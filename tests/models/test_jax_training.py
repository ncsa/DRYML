"""Focused experimental JAX Training delivery, accounting, and recovery tests."""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from threading import Event

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
_NATIVE_DEFAULT_DEVICE_PAIR = None


def _native_default_device_pairs():
    """Yield the unchanged native buffers installed only by the two-CPU child."""
    yield _NATIVE_DEFAULT_DEVICE_PAIR


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


def test_jax_dataset_continuation_native_compiled_sections(monkeypatch):
    """Real Optax/JAX qualification keeps one JIT across adjacent borrowed sections."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("optax", reason="real Optax is required for native JAX qualification")
    from tests.models.test_dataset_training_integration import _dataset_continuation_experiment
    from dryml.data import DatasetTraversal
    exp = _dataset_continuation_experiment("jax")
    captures, jit_calls = [], []
    original_capture = DatasetTraversal.capture_state
    monkeypatch.setattr(DatasetTraversal, "capture_state", lambda owner:
                        captures.append(owner.position.sequence) or original_capture(owner))
    original_jit = jax.jit
    monkeypatch.setattr(jax, "jit", lambda *a, **k: jit_calls.append(True) or original_jit(*a, **k))
    exp.train_fn(exp, callbacks=(lambda: captures == [] or pytest.fail("per-update capture"),))
    assert jit_calls == [True] and captures == [4]
    assert exp.state.input_sequence == exp.state.step == 4


def test_jax_native_transition_reuses_equal_batch_signature(monkeypatch, caplog):
    """Tiny real SGD keeps one cache entry, full validation and both barriers."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("optax")
    import dryml.models.jax.base as jax_base
    from dryml.core.state import is_reserved
    from tests.tools.benchmark_dataset_traversal import _JaxDriver
    x = np.arange(96 * 16, dtype=np.float32).reshape(96, 16) / 10000
    y = x.sum(axis=1, keepdims=True) / 8
    exp = _JaxDriver("new", x, y, sections=1).exp
    traces, signatures, cache_sizes = [], [], []
    barriers, validations, puts, update_puts = [], [], [], []
    compiled_updates = []
    original_jit = jax.jit
    original_put = jax.device_put
    original_sync = jax_base._synchronize_tree
    original_validate = exp.model._validate_candidate_state
    original_apply = exp.model._training_apply
    original_update = exp.train_fn._native_update
    prior_rng = np.asarray(jax.random.key_data(exp.model.rng)).copy()

    def device_put(value, device=None, *args, **kwargs):
        puts.append(value)
        return original_put(value, device, *args, **kwargs)

    def synchronize(*args):
        assert is_reserved(exp.model)
        barriers.append(True)
        return original_sync(*args)

    def validate(*args):
        validations.append(True)
        return original_validate(*args)

    def jit(*args, **kwargs):
        compiled = original_jit(*args, **kwargs)
        compiled_updates.append(compiled)
        return compiled

    def apply(*args):
        traces.append(True)
        return original_apply(*args)

    def native_update(*args):
        update = original_update(*args)

        def call(*values):
            assert is_reserved(exp.model)
            signatures.append([
                (leaf.shape, str(leaf.dtype), getattr(leaf, "weak_type", False), str(leaf.sharding),
                 str(getattr(leaf, "layout", None)), leaf.committed)
                for leaf in jax.tree_util.tree_leaves(values)
            ])
            before = len(puts)
            candidate = update(*values)
            update_puts.append(len(puts) - before)
            cache_sizes.append(compiled_updates[0]._cache_size())
            return candidate

        return call

    monkeypatch.setattr(exp.model, "_training_apply", apply)
    monkeypatch.setattr(exp.model, "_validate_candidate_state", validate)
    monkeypatch.setattr(jax_base, "_synchronize_tree", synchronize)
    monkeypatch.setattr(jax, "device_put", device_put)
    monkeypatch.setattr(jax, "jit", jit)
    monkeypatch.setattr(exp.train_fn, "_native_update", native_update)
    with jax.log_compiles(True):
        exp.train_fn(exp)
    assert len(traces) == 1, signatures
    assert len(compiled_updates) == 1
    compiles = [record.getMessage() for record in caplog.records
                if record.getMessage().startswith("Compiling transition ")]
    assert len(compiles) == 1, compiles
    assert cache_sizes == [1, 1, 1], signatures
    assert update_puts == [2, 0, 0]
    assert len(barriers) == len(validations) == 6
    assert exp.state.step == 3 and exp.state.examples_seen == 96
    expected = np.zeros((16, 1), dtype=np.float32)
    for start in range(0, 96, 32):
        a, b = x[start:start + 32], y[start:start + 32]
        expected -= np.float32(.001) * (2 * a.T @ (a @ expected - b) / 32)
    np.testing.assert_allclose(exp.model.parameters["kernel"], expected, rtol=1e-6)
    np.testing.assert_array_equal(jax.random.key_data(exp.model.rng), prior_rng)


@pytest.mark.exhaustive_only
def test_jax_native_uncommitted_training_respects_default_device():
    """Fresh two-CPU training preserves the original pure JIT's device inference."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    script = textwrap.dedent('''
        import json
        from unittest.mock import patch
        from dryml import F
        from dryml.core import TensorSpec
        from dryml.data import GeneratorDataset
        from dryml.models.jax import Optimizer, Training
        from tests.models import test_jax_training as fixtures
        import jax
        import numpy as np

        a, b = jax.devices()
        assert a.platform == b.platform == "cpu" and a != b

        with jax.default_device(a):
            exp = fixtures._experiment()
            exp.train_fn.optimizer = Optimizer(F("optax.sgd", 0.1))
            transformation = exp.train_fn.optimizer.bind(exp.model)
            assert jax.tree_util.tree_leaves(transformation.init(exp.model.parameters)) == []
            x = jax.numpy.ones((2, 1), dtype=jax.numpy.float32)
            y = x * 2
            fixtures._NATIVE_DEFAULT_DEVICE_PAIR = (x, y)
            spec = TensorSpec("float32", shape=(1,), batch=2, backend="jax")
            exp.train_data = GeneratorDataset(
                fixtures._native_default_device_pairs, cardinality=1, spec=(spec, spec),
            )

        original_jit = jax.jit
        original_update = Training._native_update
        candidates = []

        def native_update(*args):
            compiled = []

            def jit(transition):
                pure = original_jit(transition)
                compiled.append(pure)
                return pure

            with patch.object(jax, "jit", jit):
                update = original_update(*args)
            assert len(compiled) == 1

            def checked(*values):
                leaves = jax.tree_util.tree_leaves(values)
                assert leaves and all(not leaf.committed and leaf.device == a for leaf in leaves)
                assert values[4] is x and values[5] is y
                expected = compiled[0](*values)
                actual = update(*values)
                expected_leaves = jax.tree_util.tree_leaves(expected)
                actual_leaves = jax.tree_util.tree_leaves(actual)
                proof = {
                    "initial_device": str(a), "default_device": str(b),
                    "initial_all_uncommitted": True,
                    "pure_jit_devices": [str(leaf.device) for leaf in expected_leaves],
                    "actual_devices": [str(leaf.device) for leaf in actual_leaves],
                }
                print(json.dumps(proof), flush=True)
                assert all(leaf.device == b for leaf in expected_leaves)
                assert all(leaf.device == b for leaf in actual_leaves), proof
                for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
                    if jax.dtypes.issubdtype(actual_leaf.dtype, jax.dtypes.prng_key):
                        assert jax.random.key_impl(actual_leaf) == jax.random.key_impl(expected_leaf)
                        actual_leaf = jax.random.key_data(actual_leaf)
                        expected_leaf = jax.random.key_data(expected_leaf)
                    np.testing.assert_array_equal(actual_leaf, expected_leaf)
                candidates.append(actual)
                return actual

            return checked

        with patch.object(Training, "_native_update", staticmethod(native_update)):
            with jax.default_device(b):
                exp.train_fn(exp)
        assert len(candidates) == exp.state.step == exp.state.input_sequence == 1
        assert exp.state.examples_seen == 2
        assert exp.model.parameters["weight"].device == exp.model.rng.device == b
        assert not x.committed and x.device == a and not y.committed and y.device == a
    ''')
    environment = os.environ.copy()
    environment.update({
        "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "false",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2 --xla_cpu_multi_thread_eigen=false",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1",
    })
    result = subprocess.run(
        [sys.executable, "-c", script], env=environment, capture_output=True,
        text=True, check=False, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout, end="")


def test_jax_native_placement_failure_preserves_owners_and_frontier(monkeypatch):
    """A reserved placement failure cannot install owners or accept source input."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("optax")
    from dryml.core.state import is_reserved

    exp = _experiment()
    prior = exp.model.parameters, exp.model.mutable_state, exp.model.rng
    exp.train_fn.optimizer.bind(exp.model)
    prior_slots = exp.train_fn.optimizer.state
    original_put = jax.device_put
    failed = []

    def device_put(value, *args, **kwargs):
        if value is prior[0]["weight"]:
            assert is_reserved(exp.model)
            failed.append(True)
            raise RuntimeError("native placement failed")
        return original_put(value, *args, **kwargs)

    monkeypatch.setattr(jax, "device_put", device_put)
    with exp.train_fn.input_session(exp) as inputs:
        record = inputs.capture()
    with pytest.raises(RuntimeError, match="native placement failed"):
        exp.train_fn(exp)
    assert failed == [True] and not is_reserved(exp.model)
    assert exp.model.parameters is prior[0] and exp.model.mutable_state is prior[1]
    assert exp.model.rng is prior[2] and exp.train_fn.optimizer.state is prior_slots
    assert exp.state.step == exp.state.input_sequence == 0
    assert exp.state.dataset_state is record
    assert not exp.model._restore_failed and not exp.train_fn.optimizer._restore_failed
    with exp.train_fn.input_session(exp) as inputs:
        assert inputs.capture() is record


@pytest.mark.parametrize("implementation", ("threefry2x32", "rbg"))
def test_jax_native_placement_retains_host_backed_keys_and_weak_types(monkeypatch, implementation):
    """Native admission preserves restored key bits/implementation and weak scalars."""
    jax = pytest.importorskip("jax")
    from dryml.models.jax import Training

    key = jax.random.key(17, impl=implementation)
    host_bits = np.asarray(jax.random.key_data(key)).copy()
    host_key = jax.random.wrap_key_data(host_bits, impl=implementation)
    scalar = jax.numpy.asarray(1.)
    ready = jax.device_put(jax.numpy.ones((1,)), jax.devices()[0])
    monkeypatch.setattr(jax, "jit", lambda transition: lambda *args: args)
    update = Training._native_update(jax, None, None, None, None)
    result = update({"weight": scalar}, {}, host_key, (), ready, ready)
    placed_scalar, placed_key = result[0]["weight"], result[2]
    assert placed_scalar.committed and placed_key.committed
    assert placed_scalar.weak_type == scalar.weak_type
    assert placed_scalar.shape == scalar.shape and placed_scalar.dtype == scalar.dtype
    assert placed_scalar.sharding == scalar.sharding
    assert placed_key.shape == host_key.shape and placed_key.dtype == host_key.dtype
    assert jax.random.key_impl(placed_key) == jax.random.key_impl(host_key)
    np.testing.assert_array_equal(jax.random.key_data(placed_key), host_bits)
    assert jax.random.key_data(host_key) is host_bits
    assert result[4] is result[5] is ready


@pytest.mark.parametrize("failure", (
    "parameter_dtype", "parameter_topology", "parameter_nonfinite",
    "mutable_nonfinite", "slot_dtype", "slot_topology", "slot_nonfinite", "rng",
))
def test_jax_direct_commit_retains_complete_candidate_checks(failure):
    """Direct commits reject invalid complete candidates before changing any owner."""
    jax = pytest.importorskip("jax")
    exp = _experiment()
    exp.model.mutable_state = {"value": jax.numpy.asarray(0., dtype=jax.numpy.float32)}
    exp.train_fn.optimizer.state = {"slot": jax.numpy.asarray(0., dtype=jax.numpy.float32)}
    prior = exp.model.parameters, exp.model.mutable_state, exp.model.rng
    prior_slots = exp.train_fn.optimizer.state
    parameters, mutable, rng, slots = *prior, prior_slots
    if failure == "parameter_dtype":
        parameters = {"weight": jax.numpy.zeros((1, 1), dtype=jax.numpy.int32)}
    elif failure == "parameter_topology":
        parameters = [parameters["weight"]]
    elif failure == "parameter_nonfinite":
        parameters = {"weight": jax.numpy.full((1, 1), float("nan"))}
    elif failure == "mutable_nonfinite":
        mutable = {"value": jax.numpy.asarray(float("inf"), dtype=jax.numpy.float32)}
    elif failure == "slot_dtype":
        slots = {"slot": jax.numpy.asarray(0, dtype=jax.numpy.int32)}
    elif failure == "slot_topology":
        slots = ()
    elif failure == "slot_nonfinite":
        slots = {"slot": jax.numpy.asarray(float("nan"), dtype=jax.numpy.float32)}
    else:
        rng = jax.random.key_data(rng)
    with pytest.raises((TypeError, ValueError), match="candidate"):
        exp.train_fn._commit_update(
            exp, parameters, mutable, rng, slots,
            examples=2, loss=1., complete_epoch=False,
        )
    assert exp.model.parameters is prior[0] and exp.model.mutable_state is prior[1]
    assert exp.model.rng is prior[2] and exp.train_fn.optimizer.state is prior_slots
    assert exp.state.step == exp.state.examples_seen == 0


def test_jax_candidate_validation_keeps_shared_leaf_topology():
    """Shared identity is checked in addition to otherwise identical leaf metadata."""
    jax = pytest.importorskip("jax")
    from dryml.models.jax.base import _validate_tree_like

    shared = jax.numpy.zeros((1, 1), dtype=jax.numpy.float32)
    template = {"left": shared, "right": shared}
    _validate_tree_like(template, template, "parameters", jax)
    with pytest.raises(TypeError, match="topology"):
        _validate_tree_like(
            {"left": shared, "right": shared + 0}, template, "parameters", jax,
        )


def test_jax_native_iterator_yields_prepared_authored_batches():
    """The JAX bridge yields native batches and closes an exhausted traversal."""
    jax = pytest.importorskip("jax")
    import dryml.jax  # noqa: F401 - activate the explicit JAX adapter
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


def test_jax_invocation_telemetry_receives_host_logs_after_commits(tmp_path):
    """JAX observers run outside JIT with committed positions and host scalars."""

    pytest.importorskip("jax")
    pytest.importorskip("optax")

    repo = Repo(DirStore(tmp_path / "store"))
    exp = _experiment(repo=repo)
    logs = []

    final = exp.train(
        callbacks=[lambda values: logs.append(dict(values))],
        managed=ManagedConfig(state_repo=repo),
    ).final_state_ref

    assert [values["event"] for values in logs] == [
        "train_batch_end", "train_batch_end", "train_batch_end", "epoch_end",
    ]
    assert [values["step"] for values in logs] == [1, 2, 3, 3]
    assert [values["batch"] for values in logs[:3]] == [0, 1, 2]
    assert all(type(values["loss"]) is float for values in logs)
    assert exp.train.status(state_repo=repo).final.graph_equal(final)


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
    """Empty/infinite/noncanonical fail; unknown exact input admits without demand."""
    pytest.importorskip("jax")

    spec = TensorSpec("float32", shape=(1,), batch=1, backend="numpy")

    infinite = _experiment()
    infinite.train_data = GeneratorDataset(
        _seeded_pairs, cardinality=Cardinality.INFINITE, spec=(spec, spec), seed=7, seed_aware=True,
    )
    with pytest.raises(ValueError, match="infinite"):
        infinite.train_fn(infinite)
    assert infinite.train_fn.optimizer.obj is infinite.train_fn.optimizer.state is None

    UNKNOWN_SOURCE_OPENS.clear()
    unknown = _experiment()
    unknown.train_data = GeneratorDataset(
        _unknown_source, cardinality=Cardinality.UNKNOWN, spec=(spec, spec),
    )
    with unknown.train_fn.input_session(unknown) as inputs:
        assert unknown.state.input_recovery == "exact"
        assert inputs.traversal.position.sequence == 0
    assert UNKNOWN_SOURCE_OPENS == []
    assert unknown.train_fn.optimizer.obj is unknown.train_fn.optimizer.state is None

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
        interrupted.state._input_provider.capture()
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

    checkpoint = exp.train.status(state_repo=repo).checkpoint
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
            exp.state._input_provider.capture()
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
            exp.state._input_provider.capture()
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


def _candidate_transaction():
    """Execute the real loop transaction without importing unavailable Optax.

    This focused proof extracts the candidate-to-callback statements verbatim;
    it qualifies reservation/installation only, not native Optax training or the
    surrounding training loop. Native integration tests retain their real prerequisite.
    """
    import dryml.models.jax.base as jax_base

    function = ast.parse(textwrap.dedent(inspect.getsource(jax_base.Training._train_with_input))).body[0]
    loop = next(node for node in ast.walk(function) if isinstance(node, ast.While))
    body = loop.body[1:]
    callback_end = next(index for index, node in enumerate(body)
                        if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                        and node.target.id == "callback")
    wrapper = ast.parse(
        "def transaction(self, exp, inputs, input_candidate, update, jax, callbacks=()):\n"
        "    prepared_train = inputs.prepared\n"
        "    epoch_loss = epoch_steps = resume_batch = 0\n"
    ).body[0]
    wrapper.body.extend(body[:callback_end + 1])
    namespace = dict(vars(jax_base))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                 inspect.getsourcefile(jax_base.Training), "exec"), namespace)
    return namespace["transaction"]


def _pure_candidate(jax, parameters, mutable, rng, slots, x, y):
    """Return distinct installed JAX arrays using a local non-Optax transition."""
    del x, y
    return (jax.numpy.asarray(1.), {"weight": parameters["weight"] + 1}, mutable,
            jax.random.split(rng)[0], {"count": slots["count"] + 1})


def test_jax_candidate_reservation_precedes_shared_model_read():
    """Real-array transaction overlap rejects contention before a second pure read.

    Separate TrainStates/slot owners share one Model. Events hold the first pure
    computation open after both sessions admit; no native Optax qualification is claimed.
    """
    jax = pytest.importorskip("jax")
    from dryml.core.repo import RepoSaveError

    transaction = _candidate_transaction()
    first, second = _experiment(), _experiment()
    second = Experiment(first.model, second.train_fn, train_data=second.train_data)
    assert second.model is first.model
    assert first.state is not second.state
    for exp in (first, second):
        exp.train_fn.optimizer.state = {"count": jax.numpy.asarray(0, dtype=jax.numpy.int32)}
    admitted, start, computing, release = (Event() for _ in range(4))
    reads, callbacks = [], []

    def held_update(*args):
        reads.append("first")
        computing.set()
        assert release.wait(10), "contender did not release the pure computation"
        return _pure_candidate(jax, *args)

    def contender_update(*args):
        reads.append("second")
        return _pure_candidate(jax, *args)

    def run_first():
        with first.train_fn.input_session(first) as inputs:
            candidate = inputs.traversal.next_epoch().reserve()
            admitted.set()
            assert start.wait(10), "second input session did not admit"

            def accepted_callback():
                assert inputs._guard is None and inputs._accepted
                with inputs._repo.reserve_state_graph(first):
                    callbacks.append(first.state.step)

            transaction(first.train_fn, first, inputs, candidate, held_update, jax,
                        (accepted_callback,))

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_first)
        try:
            assert admitted.wait(10), "first input session did not admit"
            with pytest.raises(RepoSaveError, match="reserved|overlap|in progress"):
                with second.train_fn.input_session(second) as inputs:
                    candidate = inputs.traversal.next_epoch().reserve()
                    start.set()
                    assert computing.wait(10), "first candidate did not start"
                    transaction(second.train_fn, second, inputs, candidate, contender_update, jax)
        finally:
            start.set()
            release.set()
            future.result(timeout=10)

    assert reads == ["first"]
    assert callbacks == [1]
    np.testing.assert_array_equal(first.model.parameters["weight"], [[1.]])
    assert first.state.step == 1 and second.state.step == 0
    assert int(first.train_fn.optimizer.state["count"]) == 1
    assert int(second.train_fn.optimizer.state["count"]) == 0
    assert not first.model._restore_failed


@pytest.mark.parametrize("failure", ("compute", "sync", "loss", "model", "slots"))
def test_jax_pure_transaction_failures_preserve_valid_owners(monkeypatch, failure):
    """Real-array pre-installation failures release the lease without invalidation.

    The extracted transaction uses a local pure fixture, not Optax training.
    Prior input records and native references must survive without encoding.
    """
    jax = pytest.importorskip("jax")
    import dryml.models.jax.base as jax_base
    from dryml.core.state import is_reserved
    from dryml.models import TrainState

    exp = _experiment()
    exp.train_fn.optimizer.state = {"count": jax.numpy.asarray(0, dtype=jax.numpy.int32)}
    prior_parameters, prior_mutable, prior_rng = exp.model.parameters, exp.model.mutable_state, exp.model.rng
    prior_slots, prior_continuation = exp.train_fn.optimizer.state, exp.train_fn.continuation

    def update(*args):
        assert is_reserved(exp.model)
        if failure == "compute":
            raise RuntimeError("pure compute failed")
        loss, parameters, mutable, rng, slots = _pure_candidate(jax, *args)
        if failure == "loss":
            loss = jax.numpy.asarray(float("nan"))
        if failure == "model":
            mutable = {"wrong": jax.numpy.asarray(1)}
        if failure == "slots":
            slots = ()
        return loss, parameters, mutable, rng, slots

    def fail_sync(*args):
        assert is_reserved(exp.model)
        raise RuntimeError("pure sync failed")

    if failure == "sync":
        monkeypatch.setattr(jax_base, "_synchronize_tree", fail_sync)
    transaction = _candidate_transaction()
    with pytest.raises((RuntimeError, TypeError, ValueError), match="pure|finite|candidate"):
        with exp.train_fn.input_session(exp) as inputs:
            prior = inputs.capture()
            monkeypatch.setattr(TrainState, "__getstate__", lambda _: pytest.fail("candidate encoded continuation"))
            candidate = inputs.traversal.next_epoch().reserve()
            transaction(exp.train_fn, exp, inputs, candidate, update, jax)

    assert inputs._guard is None and not is_reserved(exp.model)
    assert exp.model.parameters is prior_parameters
    assert exp.model.mutable_state is prior_mutable and exp.model.rng is prior_rng
    assert exp.train_fn.optimizer.state is prior_slots
    assert exp.train_fn.continuation is prior_continuation
    assert int(prior_slots["count"]) == 0
    assert (exp.state.step, exp.state.examples_seen, exp.state.next_batch, exp.state.input_sequence) == (0, 0, 0, 0)
    assert exp.state.dataset_state is prior and inputs.traversal.position == prior.position
    assert exp.state._input_provider is None
    assert not exp.model._restore_failed and not exp.train_fn.optimizer._restore_failed
    with exp.train_fn.input_session(exp) as retry:
        assert retry.capture() is prior


@pytest.mark.parametrize("capture", (False, True))
def test_jax_callback_capture_controls_save_after_provider_retirement(tmp_path, monkeypatch, capture):
    """Only a live accepted callback capture authorizes later exact save/retry.

    This transaction proof uses local slots; native Optax recovery comparisons
    remain in the separately prerequisite-gated integration tests below.
    """
    jax = pytest.importorskip("jax")
    from dryml.core.repo import RepoSaveError
    repo = Repo(DirStore(tmp_path / "store"))
    exp = _experiment(repo=repo)
    exp.train_fn.optimizer.state = {"count": jax.numpy.asarray(0, dtype=jax.numpy.int32)}
    transaction = _candidate_transaction()

    def pause():
        provider = exp.state._input_provider
        assert provider._guard is None and provider._accepted
        assert exp.state.step == exp.state.input_sequence == 1
        if capture:
            provider.capture()
        raise RuntimeError("uncaptured callback" if not capture else "captured callback")

    with pytest.raises(RuntimeError, match="callback"):
        with exp.train_fn.input_session(exp) as inputs:
            candidate = inputs.traversal.next_epoch().reserve()
            transaction(exp.train_fn, exp, inputs, candidate,
                        lambda *args: _pure_candidate(jax, *args), jax, (pause,))

    assert exp.state._input_provider is None
    assert not exp.model._restore_failed and not exp.train_fn.optimizer._restore_failed
    assert exp.state.step == exp.state.input_sequence == 1
    if capture:
        assert exp.state.dataset_state.position == exp.state.input_position
        assert repo.save_object(exp, deep_capture=True) is not None
    else:
        assert exp.state.dataset_state is None
        with pytest.raises(RepoSaveError, match="snapshot-local state preparation failed") as error:
            repo.save_object(exp, deep_capture=True)
        assert isinstance(error.value.__cause__, ValueError)
        assert "current matching record; durable reload" in str(error.value.__cause__)
        monkeypatch.setattr(type(exp.train_data), "prepare", lambda _: pytest.fail("stale input prepared"))
        with pytest.raises(ValueError, match="current matching record.*durable reload"):
            exp.train_fn(exp)


@pytest.mark.parametrize("stage", ("before_model", "after_model", "after_optimizer", "after_accounting", "after_frontier"))
def test_jax_candidate_commit_interruptions_repair_all_owners(monkeypatch, stage):
    """Deterministic installation-seam proof; no Optax/native-training qualification."""
    jax = pytest.importorskip("jax")
    import dryml.models.jax.base as jax_base
    from dryml.models.jax import Model

    exp = _experiment()
    exp = Experiment(
        Model(F(_stateful_init_factory), F(_stateful_apply_factory), output_spec=_spec()),
        exp.train_fn, train_data=exp.train_data,
    )
    exp.train_fn.optimizer.state = {"count": jax.numpy.asarray(0, dtype=jax.numpy.int32)}
    from dryml.models import TrainState
    monkeypatch.setattr(TrainState, "__getstate__", lambda _: pytest.fail("candidate encoded continuation"))
    before = np.asarray(exp.model.parameters["weight"]).copy()
    prior_parameters, prior_mutable, prior_rng = exp.model.parameters, exp.model.mutable_state, exp.model.rng
    prior_rng_data = np.asarray(jax.random.key_data(prior_rng)).copy()
    prior_slots = exp.train_fn.optimizer.state
    candidate_mutable = {"updates": prior_mutable["updates"] + 1}
    candidate_rng = jax.random.split(prior_rng)[0]
    assert not np.array_equal(jax.random.key_data(candidate_rng), prior_rng_data)
    exp.train_fn.continuation = {"marker": jax.numpy.asarray(7, dtype=jax.numpy.int32)}

    def interrupt(observed):
        if observed == stage:
            raise RuntimeError(stage)

    monkeypatch.setattr(jax_base, "_training_commit_boundary", interrupt)
    with pytest.raises(RuntimeError, match=stage):
        with exp.train_fn.input_session(exp) as inputs:
            prior = inputs.capture()
            candidate = inputs.traversal.next_epoch().reserve()
            with inputs.update(candidate, repairable=True):
                exp.train_fn._commit_update(
                    exp, {"weight": exp.model.parameters["weight"] + 1}, candidate_mutable,
                    candidate_rng, {"count": jax.numpy.asarray(1, dtype=jax.numpy.int32)},
                    examples=2, loss=1., complete_epoch=False, input_candidate=candidate,
                )

    np.testing.assert_array_equal(exp.model.parameters["weight"], before)
    assert exp.model.parameters is prior_parameters
    assert exp.model.mutable_state is prior_mutable
    assert exp.model.rng is prior_rng
    np.testing.assert_array_equal(exp.model.mutable_state["updates"], np.asarray(0, dtype=np.int32))
    np.testing.assert_array_equal(jax.random.key_data(exp.model.rng), prior_rng_data)
    np.testing.assert_array_equal(exp.train_fn.continuation["marker"], np.asarray(7, dtype=np.int32))
    assert exp.train_fn.optimizer.state is prior_slots
    np.testing.assert_array_equal(exp.train_fn.optimizer.state["count"], np.asarray(0, dtype=np.int32))
    assert (exp.state.step, exp.state.examples_seen, exp.state.next_batch) == (0, 0, 0)
    assert exp.state.input_sequence == 0 and exp.state.dataset_state is prior
    assert inputs.traversal.position == prior.position
    assert exp.state._input_provider is None
    assert not exp.model._restore_failed and not exp.train_fn.optimizer._restore_failed
    with exp.train_fn.input_session(exp) as inputs:
        assert inputs.capture() is prior


def test_jax_candidate_failed_installation_repair_invalidates_owners(monkeypatch):
    """A local installation/repair failure retires owners instead of claiming rollback."""
    jax = pytest.importorskip("jax")
    from dryml.core.repo import RepoSaveError

    exp = _experiment()
    exp.train_fn.optimizer.state = {"count": jax.numpy.asarray(0, dtype=jax.numpy.int32)}
    install = exp.model._install_candidate_state
    calls = []

    def fail_install(*args):
        calls.append(True)
        if len(calls) == 1:
            install(*args)
            raise RuntimeError("installation failed")
        raise RuntimeError("repair failed")

    monkeypatch.setattr(exp.model, "_install_candidate_state", fail_install)
    with pytest.raises(RuntimeError, match="installation failed") as error:
        with exp.train_fn.input_session(exp) as inputs:
            inputs.capture()
            candidate = inputs.traversal.next_epoch().reserve()
            _candidate_transaction()(exp.train_fn, exp, inputs, candidate,
                                     lambda *args: _pure_candidate(jax, *args), jax)
    assert calls == [True, True]
    assert any("repair failed" in note for note in error.value.__notes__)
    assert exp.model._restore_failed and exp.train_fn.optimizer._restore_failed
    assert exp.state._input_provider is None and inputs._guard is None
    with pytest.raises(RepoSaveError, match="invalidated"):
        exp.train_fn(exp)


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
            interrupted.state._input_provider.capture()
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
        interrupted.state._input_provider.capture()
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
        interrupted.state._input_provider.capture()
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
