"""Focused persistence and lazy Optax contracts for experimental JAX owners."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

from dryml.core import Object, Repo, TensorSpec
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore


def _identity_init(key, width=1, *, scale=2.0):
    """Build one definition-configured functional parameter and mutable tree."""

    import jax

    del key
    return (
        {"scale": jax.numpy.full((width,), scale, dtype=jax.numpy.float32)},
        {"counter": jax.numpy.asarray(0, dtype=jax.numpy.int32)},
    )


def _identity_apply(parameters, mutable_state, rng_state, value, training):
    """Build a complete identity-style pure functional apply result."""

    import jax

    next_rng, _ = jax.random.split(rng_state)
    return value * parameters["scale"], {"counter": mutable_state["counter"] + int(training)}, next_rng


def _tree_apply(parameters, mutable_state, rng_state, value, training):
    """Return a valid candidate while topology tests ignore parameter values."""

    del parameters, mutable_state, training
    return value, {}, rng_state


def _list_init(key):
    """Build a list-shaped parameter template."""

    import jax

    del key
    return [jax.numpy.ones((1,), dtype=jax.numpy.float32)], {}


def _tuple_init(key):
    """Build a tuple-shaped parameter template."""

    import jax

    del key
    return (jax.numpy.ones((1,), dtype=jax.numpy.float32),), {}


def _shared_init(key):
    """Build a parameter tree with one deliberately shared leaf."""

    import jax

    del key
    shared = jax.numpy.ones((1,), dtype=jax.numpy.float32)
    return {"left": shared, "right": shared}, {}


def _duplicate_init(key):
    """Build a same-shaped parameter tree with distinct leaves."""

    import jax

    del key
    return {"left": jax.numpy.ones((1,), dtype=jax.numpy.float32), "right": jax.numpy.ones((1,), dtype=jax.numpy.float32)}, {}


def _identity_init_factory():
    """Build the test initializer from its explicit FactorySpec."""

    return _identity_init


def _identity_apply_factory():
    """Build the test apply callable from its explicit FactorySpec."""

    return _identity_apply


def _tree_apply_factory():
    """Build the topology-test apply callable."""

    return _tree_apply


def _list_init_factory():
    """Build the list-topology initializer."""

    return _list_init


def _tuple_init_factory():
    """Build the tuple-topology initializer."""

    return _tuple_init


def _shared_init_factory():
    """Build the shared-leaf topology initializer."""

    return _shared_init


def _duplicate_init_factory():
    """Build the duplicate-leaf topology initializer."""

    return _duplicate_init


def _adam_factory():
    """Build an Optax transformation with persistent moment slots."""

    import optax

    return optax.adam(0.1)


def _sgd_factory():
    """Build a deliberately different Optax recipe for rejection coverage."""

    import optax

    return optax.sgd(0.1)


def _nnx_linear_factory(*, rngs):
    """Build an NNX linear module for independent local-state coverage."""

    from flax import nnx

    return nnx.Linear(1, 1, rngs=rngs)


class ModelOptimizerGraph(Object):
    """Retain independently persisted Model and lazy Optimizer owners."""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer


def _spec():
    """Return the explicit single-output JAX specification used by tests."""

    return TensorSpec("float32", shape=(1,), backend="jax")


def _model(*, width=1, scale=2.0, seed=7, repo=None):
    """Construct a current functional Model with no prebuilt state inputs."""

    from dryml import F
    from dryml.models.jax import Model

    return Model(F(_identity_init_factory), F(_identity_apply_factory), width, scale=scale, seed=seed, output_spec=_spec(), repo=repo)


def _envelope(path: Path, name: str) -> dict:
    """Read one test-corruptible local state envelope."""

    return json.loads((path / f"{name}.json").read_text(encoding="utf-8"))


def test_model_state_round_trip_validates_current_template_and_typed_rng(tmp_path):
    """Model state restores only against the reconstructed initializer topology."""

    jax = pytest.importorskip("jax")
    from dryml.models.jax import JaxStateError

    model = _model(scale=3.0, seed=13)
    model.save_state_to_dir_imp(tmp_path, codec="pkl")
    restored = _model(scale=0.0, seed=13)
    restored.restore_state_from_dir_imp(tmp_path, codec="pkl")
    np.testing.assert_array_equal(restored.parameters["scale"], np.asarray([3.0], dtype=np.float32))
    assert str(restored.rng.dtype).startswith("key<")

    wrong_shape = _model(width=2)
    with pytest.raises(JaxStateError, match="topology"):
        wrong_shape.restore_state_from_dir_imp(tmp_path, codec="pkl")
    assert jax.random.key_data(wrong_shape.rng).shape == jax.random.key_data(model.rng).shape


@pytest.mark.parametrize("corruption", ("version", "path", "dtype", "missing_payload"))
def test_functional_state_corruption_rejects_without_partial_install(tmp_path, corruption):
    """Malformed model envelope data leaves the prior reconstructed state intact."""

    jax = pytest.importorskip("jax")
    from dryml.models.jax import JaxStateError

    saved = _model(scale=9.0)
    saved.save_state_to_dir_imp(tmp_path, codec="pkl")
    target = _model(scale=4.0)
    before = np.asarray(target.parameters["scale"]).copy()
    envelope = _envelope(tmp_path, "model-state")
    if corruption == "version":
        envelope["version"] = 99
    elif corruption == "path":
        envelope["leaves"][0]["path"] = ["bad:path"]
    elif corruption == "dtype":
        envelope["leaves"][0]["dtype"] = "float64"
    else:
        (tmp_path / "model-state-0.npy").unlink()
    if corruption != "missing_payload":
        (tmp_path / "model-state.json").write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(JaxStateError):
        target.restore_state_from_dir_imp(tmp_path, codec="pkl")
    np.testing.assert_array_equal(target.parameters["scale"], before)


def test_functional_state_rejects_container_and_shared_leaf_topology_changes(tmp_path):
    """Same-shaped trees still reject list/tuple and shared/duplicate mismatches."""

    pytest.importorskip("jax")
    from dryml import F
    from dryml.models.jax import JaxStateError, Model

    source = Model(F(_list_init_factory), F(_tree_apply_factory), output_spec=_spec())
    source.save_state_to_dir_imp(tmp_path, codec="pkl")
    with pytest.raises(JaxStateError, match="topology"):
        Model(F(_tuple_init_factory), F(_tree_apply_factory), output_spec=_spec()).restore_state_from_dir_imp(tmp_path, codec="pkl")

    alias_dir = tmp_path / "alias"
    alias_dir.mkdir()
    Model(F(_shared_init_factory), F(_tree_apply_factory), output_spec=_spec()).save_state_to_dir_imp(alias_dir, codec="pkl")
    with pytest.raises(JaxStateError, match="topology"):
        Model(F(_duplicate_init_factory), F(_tree_apply_factory), output_spec=_spec()).restore_state_from_dir_imp(alias_dir, codec="pkl")


def test_optimizer_is_unbound_until_training_seam_binds_current_parameters(tmp_path):
    """Constructing/saving an inference graph does not initialize or import Optax."""

    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from dryml import F
    from dryml.models.jax import Optimizer

    optimizer = Optimizer(F(_adam_factory))
    assert optimizer.obj is None
    assert optimizer.state is None
    optimizer.save_state_to_dir_imp(tmp_path, codec="pkl")
    assert _envelope(tmp_path, "optimizer-owner")["bound"] is False

    model = _model()
    transformation = optimizer.bind(model)
    assert transformation is optimizer.obj
    assert optimizer.state is not None


def test_bound_optimizer_restores_recipe_slots_against_current_template(tmp_path):
    """A bound restore reconstructs from recipe only when the matching Model binds."""

    jax = pytest.importorskip("jax")
    pytest.importorskip("optax")
    from dryml import F
    from dryml.models.jax import Optimizer

    model = _model()
    optimizer = Optimizer(F(_adam_factory))
    optimizer.bind(model)
    _, optimizer.state = optimizer.obj.update(
        {"scale": jax.numpy.ones((1,), dtype=jax.numpy.float32)}, optimizer.state, model.parameters,
    )
    optimizer.save_state_to_dir_imp(tmp_path, codec="pkl")

    restored = Optimizer(F(_adam_factory))
    restored.restore_state_from_dir_imp(tmp_path, codec="pkl")
    assert restored.obj is None
    restored.bind(_model())
    assert np.asarray(restored.state[0].count).item() == 1

    incompatible = Optimizer(F(_adam_factory))
    incompatible.restore_state_from_dir_imp(tmp_path, codec="pkl")
    with pytest.raises(ValueError, match="incompatible model parameter template"):
        incompatible.bind(_model(width=2))


def test_bound_optimizer_restore_owns_pending_payload_after_source_is_removed(tmp_path):
    """Deferred model binding never borrows a Store payload directory lifetime."""

    jax = pytest.importorskip("jax")
    pytest.importorskip("optax")
    from dryml import F
    from dryml.models.jax import Optimizer

    payload = tmp_path / "payload"
    payload.mkdir()
    optimizer = Optimizer(F(_adam_factory))
    model = _model()
    optimizer.bind(model)
    _, optimizer.state = optimizer.obj.update(
        {"scale": jax.numpy.ones((1,), dtype=jax.numpy.float32)},
        optimizer.state,
        model.parameters,
    )
    optimizer.save_state_to_dir_imp(payload, codec="pkl")

    restored = Optimizer(F(_adam_factory))
    restored.restore_state_from_dir_imp(payload, codec="pkl")
    payload.rename(tmp_path / "moved")

    restored.bind(_model())
    assert np.asarray(restored.state[0].count).item() == 1


def test_optimizer_factory_identity_rejects_before_slot_decode(tmp_path):
    """A different Optax recipe cannot decode or update retained slot state."""

    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from dryml import F
    from dryml.models.jax import Optimizer

    optimizer = Optimizer(F(_adam_factory))
    optimizer.bind(_model())
    optimizer.save_state_to_dir_imp(tmp_path, codec="pkl")
    incompatible = Optimizer(F(_sgd_factory))
    with pytest.raises(ValueError, match="incompatible factory"):
        incompatible.restore_state_from_dir_imp(tmp_path, codec="pkl")
    assert incompatible.obj is None

    optimizer.factory = F(_sgd_factory)
    with pytest.raises(ValueError, match="identity changed"):
        optimizer.bind(_model())


def test_repo_round_trip_keeps_model_and_optimizer_owners_independent(tmp_path):
    """Graph restore retains lazy binding evidence without a live transformation."""

    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from dryml import F
    from dryml.models.jax import Optimizer

    repo = Repo(DirStore(tmp_path / "store"))
    model = _model(repo=repo)
    optimizer = Optimizer(F(_adam_factory), repo=repo)
    optimizer.bind(model)
    state = repo.save_object(ModelOptimizerGraph(model, optimizer, repo=repo), deep_capture=True)
    loaded = Repo(DirStore.open_existing(tmp_path / "store")).load_state_ref(state, reuse_live="never")

    assert loaded.optimizer.obj is None
    loaded.optimizer.bind(loaded.model)
    assert loaded.optimizer.state is not None


def test_nnx_state_round_trip_partitions_param_and_mutable_values(tmp_path):
    """NNX reconstruction validates factory topology before installing values."""

    pytest.importorskip("jax")
    pytest.importorskip("flax.nnx")
    from dryml import F
    from dryml.models.jax import NNXModel

    model = NNXModel(F(_nnx_linear_factory), output_spec=_spec(), seed=3)
    model.obj.kernel.value = model.obj.kernel.value * 0.0 + 5.0
    model.save_state_to_dir_imp(tmp_path, codec="pkl")
    restored = NNXModel(F(_nnx_linear_factory), output_spec=_spec(), seed=3)
    restored.restore_state_from_dir_imp(tmp_path, codec="pkl")
    np.testing.assert_array_equal(restored.obj.kernel.value, np.asarray([[5.0]], dtype=np.float32))


def test_nnx_state_rejects_corrupt_topology_without_install(tmp_path):
    """NNX parameter/mutable partition corruption cannot partly update a module."""

    pytest.importorskip("jax")
    pytest.importorskip("flax.nnx")
    from dryml import F
    from dryml.models.jax import JaxStateError, NNXModel

    model = NNXModel(F(_nnx_linear_factory), output_spec=_spec())
    model.save_state_to_dir_imp(tmp_path, codec="pkl")
    target = NNXModel(F(_nnx_linear_factory), output_spec=_spec())
    before = np.asarray(target.obj.kernel.value).copy()
    envelope = _envelope(tmp_path, "model-state")
    envelope["tree"] = "corrupt NNX topology"
    (tmp_path / "model-state.json").write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(JaxStateError, match="topology"):
        target.restore_state_from_dir_imp(tmp_path, codec="pkl")
    np.testing.assert_array_equal(target.obj.kernel.value, before)


def test_repo_restore_invalidates_existing_graph_after_model_state_failure(tmp_path):
    """A later local topology failure invalidates live graph objects, not Store data."""

    jax = pytest.importorskip("jax")
    repo = Repo(DirStore(tmp_path / "store"))
    model = _model(repo=repo)
    state = repo.save_object(model, deep_capture=True)
    model.parameters = {"different": jax.numpy.zeros((1,), dtype=jax.numpy.float32)}
    with pytest.raises(RepoLoadError, match="Targeted exact restore failed"):
        repo.restore_state_ref_into(model, state)
    assert model._restore_failed
    fresh = Repo(DirStore.open_existing(tmp_path / "store")).load_state_ref(state, reuse_live="never")
    np.testing.assert_array_equal(fresh.parameters["scale"], np.asarray([2.0], dtype=np.float32))


def test_fresh_process_inference_only_load_never_imports_optax(tmp_path):
    """A Model-only graph saves and loads while imports explicitly reject Optax."""

    pytest.importorskip("jax")
    script = """
import base64
import builtins
import pickle
import sys
from dryml import F
from dryml.core import Repo, TensorSpec
from dryml.core.store.dir import DirStore
from dryml.core.symbol import SourceSpec

original_import = builtins.__import__
def no_optax(name, *args, **kwargs):
    if name == 'optax' or name.startswith('optax.'):
        raise AssertionError('Optax must not be imported for inference-only Model state')
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_optax
from dryml.models.jax import Model
import jax
init = SourceSpec.from_source("lambda: lambda key: ({'scale': __import__('jax').numpy.asarray([3.0], dtype=__import__('jax').numpy.float32)}, {'counter': __import__('jax').numpy.asarray(0, dtype=__import__('jax').numpy.int32)})")
apply = SourceSpec.from_source("lambda: lambda parameters, mutable_state, rng_state, value, training: (value * parameters['scale'], mutable_state, rng_state)")
if sys.argv[1] == 'save':
    repo = Repo(DirStore(sys.argv[2]))
    model = Model(F(init), F(apply), output_spec=TensorSpec('float32', shape=(1,), backend='jax'), repo=repo)
    state = repo.save_object(model, deep_capture=True)
    print(base64.b64encode(pickle.dumps(state)).decode('ascii'))
else:
    state = pickle.loads(base64.b64decode(sys.argv[3]))
    model = Repo(DirStore.open_existing(sys.argv[2])).load_state_ref(state, reuse_live='never')
    assert model(jax.numpy.asarray([2.0], dtype=jax.numpy.float32)).tolist() == [6.0]
"""
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[2] / "src"), JAX_PLATFORMS="cpu")
    store = tmp_path / "store"
    saved = subprocess.run([sys.executable, "-c", script, "save", str(store)], capture_output=True, text=True, env=environment, check=False)
    assert saved.returncode == 0, saved.stderr
    loaded = subprocess.run([sys.executable, "-c", script, "load", str(store), saved.stdout.strip()], capture_output=True, text=True, env=environment, check=False)
    assert loaded.returncode == 0, loaded.stderr
