"""Focused contracts for experimental JAX and Flax NNX model ownership."""

from __future__ import annotations

import numpy as np
import pytest

from dryml.core import TensorSpec


def _linear_init(key, width, *, scale=1.0):
    """Initialize a linear parameter tree and mutable call statistic."""

    import jax

    return (
        {"kernel": jax.numpy.full((width, 1), scale, dtype=jax.numpy.float32)},
        {"calls": jax.numpy.asarray(0, dtype=jax.numpy.int32)},
    )


def _linear_apply(parameters, mutable_state, rng_state, value, training):
    """Return a complete functional apply candidate without parameter updates."""

    import jax

    next_rng, _ = jax.random.split(rng_state)
    return (
        value @ parameters["kernel"],
        {"calls": mutable_state["calls"] + int(training) + 1},
        next_rng,
    )


def _linear_init_factory():
    """Build the functional initializer selected by ``F(...)``."""

    return _linear_init


def _linear_apply_factory():
    """Build the functional apply callable selected by ``F(...)``."""

    return _linear_apply


def _bad_initializer(key):
    """Return an intentionally malformed initializer result."""

    del key
    return {}


def _bad_state_initializer(key):
    """Return unsupported plain functional mutable state."""

    import jax

    del key
    return {"weight": jax.numpy.ones((1,))}, {"bad": "state"}


def _bad_apply(parameters, mutable_state, rng_state, value, training):
    """Return an intentionally malformed apply result."""

    del parameters, mutable_state, rng_state, value, training
    return None


def _bad_initializer_factory():
    """Build a malformed initializer for protocol rejection coverage."""

    return _bad_initializer


def _bad_state_initializer_factory():
    """Build an initializer with malformed mutable state."""

    return _bad_state_initializer


def _bad_apply_factory():
    """Build a malformed functional apply callable."""

    return _bad_apply


def _nnx_linear_factory(*, rngs):
    """Build a minimal NNX linear module."""

    from flax import nnx

    return nnx.Linear(2, 1, rngs=rngs)


def _nnx_batch_norm_factory(*, rngs):
    """Build an NNX module with train-time mutable batch statistics."""

    from flax import nnx

    return nnx.BatchNorm(1, use_running_average=False, rngs=rngs)


def _spec(shape=(1,)):
    """Return the explicit JAX output specification used by these wrappers."""

    return TensorSpec("float32", shape=shape, backend="jax")


def _model(*, width=2, scale=1.0, seed=7, trainable_mask=None):
    """Construct one functional test Model using the accepted constructor API."""

    from dryml import F
    from dryml.models.jax import Model

    return Model(
        F(_linear_init_factory), F(_linear_apply_factory), width,
        scale=scale,
        seed=seed,
        trainable_mask=trainable_mask,
        output_spec=_spec(),
    )


def test_functional_constructor_records_initializer_arguments_and_splits_seed():
    """Initialization receives authored arguments and never retains its key for use."""

    jax = pytest.importorskip("jax")
    model = _model(width=3, scale=2.5, seed=19)
    init_key, expected_next = jax.random.split(jax.random.key(19))

    assert model.init_args == (3,)
    assert model.init_kwargs == {"scale": 2.5}
    assert np.asarray(model.parameters["kernel"]).tolist() == [[2.5], [2.5], [2.5]]
    np.testing.assert_array_equal(jax.random.key_data(model.rng), jax.random.key_data(expected_next))
    assert not np.array_equal(jax.random.key_data(init_key), jax.random.key_data(model.rng))


def test_functional_model_supports_raw_selected_map_and_autoencoder_calls():
    """One prediction-only contract supports raw, selected, Map, and AutoEncoder."""

    jax = pytest.importorskip("jax")
    from dryml.data import ArrayDataset, Batch, Map
    from dryml.models import AutoEncoder

    model = _model(scale=2.0)
    element_spec = TensorSpec("float32", shape=(2,), backend="jax")
    batch_spec = element_spec.with_batch(2)

    assert np.asarray(model(jax.numpy.asarray([[1.0, 2.0]]))).tolist() == [[6.0]]
    assert np.asarray(model.find_implementation(element_spec)(jax.numpy.asarray([1.0, 2.0]))).tolist() == [6.0]
    assert np.asarray(model.find_implementation(batch_spec)(jax.numpy.asarray([[1.0, 2.0], [2.0, 1.0]]))).tolist() == [[6.0], [6.0]]

    encoder = _model(width=1, scale=2.0)
    decoder = _model(width=1, scale=3.0)
    mapped = Map(ArrayDataset(np.asarray([[1.0], [2.0]], dtype=np.float32)), encoder)
    autoencoded = Map(Batch(ArrayDataset(np.asarray([[1.0], [2.0]], dtype=np.float32)), 2), AutoEncoder(encoder, decoder))
    assert [np.asarray(value).tolist() for value in mapped] == [[2.0], [4.0]]
    assert [np.asarray(value).tolist() for value in autoencoded] == [[[6.0], [12.0]]]


def test_functional_initializer_and_apply_results_are_validated():
    """Malformed protocol results fail before any prediction state is exposed."""

    pytest.importorskip("jax")
    from dryml import F
    from dryml.models.jax import Model

    with pytest.raises(TypeError, match="initializer"):
        Model(F(_bad_initializer_factory), F(_linear_apply_factory), output_spec=_spec())
    with pytest.raises(TypeError, match="mutable_state"):
        Model(F(_bad_state_initializer_factory), F(_linear_apply_factory), output_spec=_spec())
    model = Model(F(_linear_init_factory), F(_bad_apply_factory), 1, output_spec=_spec())
    with pytest.raises(TypeError, match="exactly"):
        model(np.asarray([[1.0]], dtype=np.float32))


def test_public_prediction_discards_functional_mutable_and_rng_candidates():
    """Evaluation is a snapshot even when a candidate would advance both values."""

    jax = pytest.importorskip("jax")
    model = _model()
    before_calls = np.asarray(model.mutable_state["calls"]).copy()
    before_rng = jax.random.key_data(model.rng).copy()

    predictions, candidate_state, candidate_rng = model._candidate_apply(
        jax.numpy.ones((1, 2)), training=True,
    )
    assert tuple(predictions.shape) == (1, 1)
    assert np.asarray(candidate_state["calls"]).item() == 2
    assert not np.array_equal(jax.random.key_data(candidate_rng), before_rng)
    model(jax.numpy.ones((1, 2)))
    np.testing.assert_array_equal(model.mutable_state["calls"], before_calls)
    np.testing.assert_array_equal(jax.random.key_data(model.rng), before_rng)


def test_output_spec_is_required_and_inferred_without_forward_probe():
    """Output metadata is explicit and never executes the functional apply callable."""

    pytest.importorskip("jax")
    from dryml import F
    from dryml.models.jax import Model

    with pytest.raises(TypeError, match="output_spec"):
        Model(F(_linear_init_factory), F(_linear_apply_factory), 1)
    model = Model(
        F(_linear_init_factory), F(_bad_apply_factory), 1, output_spec=_spec(),
    )
    assert model.infer_output_spec(
        TensorSpec("float32", shape=(1,), backend="jax")
    ) == _spec()


def test_state_restore_refreshes_frozen_trainable_references(tmp_path):
    """Restoration never leaves measurement pointing at pre-restore arrays."""

    jax = pytest.importorskip("jax")
    from dryml.models import ParameterCounts

    source = _model(width=2, trainable_mask={"kernel": False})
    source.parameters = {"kernel": jax.numpy.ones((2, 1), dtype=jax.numpy.float32) * 9}
    source.save_state_to_dir_imp(tmp_path, codec="pkl")
    target = _model(width=2, trainable_mask={"kernel": False})
    prior = target.parameters["kernel"]
    target.restore_state_from_dir_imp(tmp_path, codec="pkl")

    assert target.parameters["kernel"] is not prior
    assert target.trainable_parameters() == ()
    assert target.parameter_counts() == ParameterCounts(total=2, trainable=0)


def test_nnx_prediction_discards_mutable_candidate_and_keeps_external_rng_distinct():
    """NNX snapshot calls do not commit BatchNorm/RNG candidates to the module."""

    jax = pytest.importorskip("jax")
    pytest.importorskip("flax.nnx")
    from dryml import F
    from dryml.models.jax import NNXModel

    model = NNXModel(F(_nnx_batch_norm_factory), output_spec=_spec(), seed=11)
    init_key, expected_rng = jax.random.split(jax.random.key(11))
    before = model._state_tree()
    _, candidate_mutable, candidate_rng = model._candidate_apply(
        jax.numpy.asarray([[1.0], [3.0]], dtype=jax.numpy.float32), training=True,
    )
    assert not np.array_equal(np.asarray(candidate_mutable["mean"].value), np.asarray(before["mutable_state"]["mean"].value))
    assert not np.array_equal(jax.random.key_data(candidate_rng), jax.random.key_data(model.rng))
    model(jax.numpy.asarray([[1.0], [3.0]], dtype=jax.numpy.float32))
    np.testing.assert_array_equal(model.obj.mean.value, before["mutable_state"]["mean"].value)
    np.testing.assert_array_equal(jax.random.key_data(model.rng), jax.random.key_data(expected_rng))
    assert not np.array_equal(jax.random.key_data(init_key), jax.random.key_data(model.rng))
    with pytest.raises(ValueError, match="reserved 'rngs'"):
        NNXModel(F(_nnx_linear_factory, rngs="caller"), output_spec=_spec())
