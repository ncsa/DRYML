import gc
import warnings
import weakref

import numpy as np
import pytest

from dryml import F
from dryml.core import Object, Repo
from dryml.core.query import field
from dryml.core.store.dir import DirStore
from dryml.core.tensor_spec import TensorSpec
from dryml.data import ArgMax, ArrayDataset, Map, Pipe, Project, Select
from dryml.managed import ManagedConfig
from dryml.models import AutoEncoder, Experiment, TrainState


tf = pytest.importorskip("tensorflow")


class TinyKerasModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, x):
        return self.dense(x)


class EffectfulKerasModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def call(self, x):
        self.calls += 1
        return x


class ZeroKerasModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer="zeros")

    def call(self, x):
        return self.dense(x)


class NestedDynamicRegularizer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.normalizer = tf.keras.layers.BatchNormalization()

    def call(self, value, training=None):
        result = self.normalizer(value, training=training)
        self.add_loss(tf.cast(0.25, result.dtype))
        return result


class NestedDynamicKerasModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer="ones")
        self.regularizer = NestedDynamicRegularizer()

    def call(self, value, training=None):
        return self.regularizer(self.dense(value), training=training)


class FactoryTopology(Object):
    """Retain two references to one factory-built model for topology checks."""

    def __init__(self, primary, mirror):
        self.primary = primary
        self.mirror = mirror


def test_tf_basic_training_updates_experiment_state():
    from dryml.models.tf import BasicTraining, Loss, Model, Optimizer

    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32)
    ds = ArrayDataset((x, y))

    model = Model(TinyKerasModel)
    optimizer = Optimizer(tf.keras.optimizers.SGD, learning_rate=0.01)
    loss = Loss(tf.keras.losses.MeanSquaredError)
    train_fn = BasicTraining(optimizer=optimizer, loss=loss, epochs=1, batch_size=2, verbose=0)
    exp = Experiment(model, train_fn, train_data=ds)

    history = train_fn(exp)

    assert model.obj is model.mdl
    assert optimizer.obj is not None
    assert history is not None
    assert exp.state.epoch == 1
    assert exp.state.step == 2
    assert exp.state.examples_seen == 4
    assert exp.state.phase is None
    assert float(optimizer.obj.learning_rate.numpy()) == pytest.approx(0.01)


def test_tf_managed_experiment_smoke_returns_its_terminal_receipt(tmp_path):
    """A tiny CPU-managed Keras run publishes one terminal Experiment receipt."""

    from dryml.models.tf import BasicTraining, Loss, Model, Optimizer

    repo = Repo(DirStore(tmp_path / "store"))
    dataset = ArrayDataset((
        np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
        np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32),
    ))
    model = Model(TinyKerasModel)
    train_fn = BasicTraining(
        optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.01),
        loss=Loss(tf.keras.losses.MeanSquaredError), epochs=1, batch_size=2, verbose=0,
    )
    exp = Experiment(model, train_fn, train_data=dataset, repo=repo)

    final = exp.train(managed=ManagedConfig(state_repo=repo))

    assert final == exp.train.status(state_repo=repo).final_state_ref
    assert exp.state.phase == TrainState.trained
    assert exp.state.examples_seen == 4


def test_tf_model_and_optimizer_state_ref_round_trip(tmp_path):
    from dryml.models.tf import BasicTraining, Loss, Model, Optimizer

    repo = Repo(stores=tmp_path)
    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32)
    ds = ArrayDataset((x, y), repo=repo)
    model = Model(TinyKerasModel, repo=repo)
    optimizer = Optimizer(
        tf.keras.optimizers.SGD,
        learning_rate=0.01,
        momentum=0.9,
        repo=repo,
    )
    train_fn = BasicTraining(
        optimizer=optimizer,
        loss=Loss(tf.keras.losses.MeanSquaredError, repo=repo),
        epochs=1,
        batch_size=2,
        verbose=0,
        repo=repo,
    )
    exp = Experiment(model, train_fn, train_data=ds, repo=repo)
    exp.train_fn(exp)
    expected_predictions = model.obj(tf.convert_to_tensor(x)).numpy()
    expected_iterations = int(optimizer.obj.iterations.numpy())
    expected_optimizer = [value.numpy().copy() for value in optimizer.obj.variables]

    state = repo.save_object(exp, deep_capture=True)
    repo.close(flush=True)
    loaded = Repo(stores=tmp_path).load_state_ref(state, reuse_live="never")
    assert loaded.state.examples_seen == 4
    assert loaded.state.next_batch == 0
    loaded_predictions = loaded.model(tf.convert_to_tensor(x)).numpy()
    loaded_optimizer_wrapper = loaded.train_fn.optimizer
    loaded_optimizer = loaded_optimizer_wrapper.obj
    loaded_optimizer.build(loaded.model.obj.trainable_variables)
    loaded_optimizer_wrapper.restore_pending()

    np.testing.assert_allclose(loaded_predictions, expected_predictions)
    assert int(loaded_optimizer.iterations.numpy()) == expected_iterations
    assert len(loaded_optimizer.variables) == len(expected_optimizer)
    for actual, expected in zip(loaded_optimizer.variables, expected_optimizer):
        np.testing.assert_allclose(actual.numpy(), expected)

    loaded.model.obj.trainable_variables[0].assign_add(
        tf.ones_like(loaded.model.obj.trainable_variables[0])
    )
    changed_predictions = loaded.model(tf.convert_to_tensor(x)).numpy()
    assert not np.allclose(changed_predictions, expected_predictions)
    loaded_optimizer.iterations.assign_add(1)
    assert loaded_optimizer_wrapper.restore_pending() is None
    assert int(loaded_optimizer.iterations.numpy()) == expected_iterations + 1

    rebound = Repo(stores=tmp_path).load_state_ref(state, reuse_live="never")
    selected = rebound.model.find_implementation(input_spec=ArrayDataset(x).spec)
    first = selected(x[0])
    np.testing.assert_allclose(first.numpy(), expected_predictions[0])


def test_tf_stateless_wrapper_publishes_empty_payload(tmp_path):
    from dryml.models.tf import Wrapper

    repo = Repo(stores=tmp_path)
    wrapper = Wrapper(object, repo=repo)

    state = repo.save_object(wrapper)
    loaded = Repo(stores=tmp_path).load_state_ref(state, reuse_live="never")

    assert type(loaded.obj) is object


def test_tf_low_level_training_resumes_model_and_optimizer_state(tmp_path):
    from dryml.models.tf import Loss, Model, Optimizer, Training

    repo = Repo(stores=tmp_path)
    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32)
    model = Model(TinyKerasModel, repo=repo)
    optimizer = Optimizer(
        tf.keras.optimizers.SGD,
        learning_rate=0.01,
        momentum=0.9,
        repo=repo,
    )
    exp = Experiment(
        model,
        Training(
            optimizer=optimizer,
            loss=Loss(tf.keras.losses.MeanSquaredError, repo=repo),
            epochs=1,
            batch_size=2,
            verbose=0,
            repo=repo,
        ),
        train_data=ArrayDataset((x, y), repo=repo),
        repo=repo,
    )
    exp.train_fn(exp)
    state = repo.save_object(exp, deep_capture=True)

    exp.train_fn(exp)
    expected_predictions = model(tf.convert_to_tensor(x)).numpy()
    expected_optimizer = [value.numpy().copy() for value in optimizer.obj.variables]

    loaded = Repo(stores=tmp_path).load_state_ref(state, reuse_live="never")
    loaded.train_fn(loaded)
    loaded_predictions = loaded.model(tf.convert_to_tensor(x)).numpy()
    loaded_optimizer = loaded.train_fn.optimizer.obj.variables

    np.testing.assert_allclose(loaded_predictions, expected_predictions)
    assert len(loaded_optimizer) == len(expected_optimizer)
    for actual, expected in zip(loaded_optimizer, expected_optimizer):
        np.testing.assert_allclose(actual.numpy(), expected)


def test_tf_sequential_infers_output_spec_without_explicit_output_spec():
    from dryml.models.tf import Sequential

    x = np.zeros((4, 3), dtype=np.float32)
    ds = ArrayDataset(x)
    model = Sequential(layer_defs=(F("Dense", units=2),))

    assert Map(ds, model).spec == TensorSpec("float32", shape=(2,), backend="tf")


def test_tf_custom_dtype_layer_requires_explicit_output_spec_without_execution():
    from dryml.models.tf import Model

    class CastLayer(tf.keras.layers.Layer):
        calls = 0

        def call(self, inputs):
            self.calls += 1
            return tf.cast(inputs, tf.int32)

        def compute_output_shape(self, input_shape):
            return input_shape

    model = object.__new__(Model)
    model.obj = tf.keras.Sequential([CastLayer()])
    model.output_spec = None

    with pytest.raises(NotImplementedError, match="pass output_spec explicitly"):
        model.infer_output_spec(TensorSpec("float32", shape=(3,), backend="tf"))
    assert model.obj.layers[0].calls == 0


def test_tf_inference_never_invokes_an_opaque_model_or_changes_mode():
    from dryml.models.tf import Model

    model = Model(EffectfulKerasModel)
    model.obj.trainable = False

    with pytest.raises(NotImplementedError, match="pass output_spec explicitly"):
        model.infer_output_spec(TensorSpec("float32", shape=(3,), backend="numpy"))

    assert model.obj.calls == 0
    assert model.obj.trainable is False


def test_tf_selected_element_and_batched_calls_preserve_batch_boundaries_and_cache():
    from dryml.models.tf import Sequential

    model = Sequential(layer_defs=(F("Dense", units=2),))
    element_spec = TensorSpec("float32", shape=(3,), backend="numpy")
    batch_spec = TensorSpec("float32", shape=(3,), batch=2, backend="numpy")
    element = model.find_implementation(input_spec=element_spec)
    batched = model.find_implementation(input_spec=batch_spec)

    assert tuple(element(np.zeros((3,), dtype=np.float32)).shape) == (2,)
    assert tuple(batched(np.zeros((2, 3), dtype=np.float32)).shape) == (2, 2)

    model.learn()
    direct = model(tf.zeros((1, 3), dtype=tf.float32))
    assert model.call_mode == "cached"
    assert tuple(model(tf.zeros((1, 3), dtype=tf.float32)).shape) == tuple(direct.shape)
    model.eager()
    assert model.call_mode == "eager"


def test_tf_cached_element_invoker_does_not_retain_the_model():
    """The weak preparation table releases a model with selected element adaptation."""

    from dryml.models.tf import Sequential

    model = Sequential(layer_defs=(F("Dense", units=2),))
    model.default_batched = False
    model.learn()
    model(np.zeros((3,), dtype=np.float32))
    reference = weakref.ref(model)

    del model
    gc.collect()

    assert reference() is None


def test_tf_sequential_accepts_explicit_factory_specs():
    from dryml.models.tf import Sequential

    x = np.zeros((4, 32, 32, 1), dtype=np.float32)
    ds = ArrayDataset(x)
    encoder = Sequential(layer_defs=[
        F("Flatten"),
        F("Dense", 32, activation="relu"),
        F("Dense", 2, activation="linear"),
    ])

    decoder = Sequential(layer_defs=[
        F("Dense", 32 * 32, activation="linear"),
        F("Reshape", (32, 32, 1)),
    ])

    assert Map(ds, encoder).spec == TensorSpec("float32", shape=(2,), backend="tf")
    assert decoder.infer_output_spec(TensorSpec("float32", shape=(2,), backend="tf")) == TensorSpec(
        "float32",
        shape=(32, 32, 1),
        backend="tf",
    )


def test_tf_sequential_rejects_shorthand_before_constructing_any_layer(monkeypatch):
    from dryml.models.tf import Sequential

    built = []

    def build(layer_def, **kwargs):
        built.append(layer_def)
        return tf.keras.layers.ReLU()

    monkeypatch.setattr(type(F("ReLU")), "build", build)
    for layer_defs in (["ReLU"], [("Dense", 2)], [["Dense", 2]]):
        with pytest.raises(TypeError, match="explicit FactorySpec.*Use F"):
            Sequential(layer_defs=layer_defs)
    with pytest.raises(TypeError, match="explicit FactorySpec.*Use F"):
        Sequential(layer_defs=[F("ReLU"), "ReLU"])

    assert built == []


def test_tf_sequential_rejects_wrong_backend_layer_type():
    from dryml.models.tf import Sequential

    with pytest.raises(TypeError, match="FactorySpec built object"):
        Sequential(layer_defs=[F(object)])


def test_tf_sequential_factory_state_ref_round_trip(tmp_path):
    from dryml.models.tf import Sequential

    repo = Repo(stores=tmp_path)
    model = Sequential(
        layer_defs=[F("Dense", 1, use_bias=False, kernel_initializer="ones")],
        repo=repo,
    )
    value = tf.constant([[2.0]], dtype=tf.float32)
    expected = model(value).numpy()
    root = FactoryTopology(model, model, repo=repo)
    peer = Sequential(
        layer_defs=[F("Dense", 1, use_bias=False, kernel_initializer="zeros")],
        repo=repo,
    )
    peer(value)

    state = repo.save_object(root, deep_capture=True)
    repo.save_object(FactoryTopology(peer, peer, repo=repo), deep_capture=True)
    repo.set_metadata(state.object, {"scenario": "explicit-factory"})
    repo.close(flush=True)
    reopened = Repo(stores=tmp_path)
    state_hash = next(iter(state.states.values()))

    assert reopened.references().exact(state.object).state_hash(state_hash).where(
        field("object", "scenario").eq("explicit-factory")
    ).state_refs().one() == state
    assert list(
        reopened.query(root.definition)
        .categorical(drop=("layer_defs",), recursive=True)
        .exact(path="primary")
        .stored()
        .defs()
    ) == [root.definition]

    loaded = reopened.load_state_ref(state, reuse_live="never")

    np.testing.assert_allclose(loaded.primary(value).numpy(), expected)
    assert loaded.primary is loaded.mirror
    assert loaded.object_ref == state.object
    assert loaded.last_state_ref == state


def test_tf_model_map_unbatched_image_uses_backend_batch_axis():
    from dryml.models.tf import Sequential

    x = np.zeros((2, 28, 28, 1), dtype=np.float32)
    ds = ArrayDataset(x)
    model = Sequential(layer_defs=[
        F("Flatten"),
        F("Dense", 2),
    ])
    mapped = Map(ds, model)

    out = list(mapped)

    assert mapped.spec == TensorSpec("float32", shape=(2,), backend="tf")
    assert [tuple(item.shape) for item in out] == [(2,), (2,)]


def test_tf_model_project_pipe_maps_unbatched_images_and_preserves_labels():
    from dryml.models.tf import Sequential

    x = np.zeros((2, 28, 28, 1), dtype=np.float32)
    y = np.array([3, 7], dtype=np.int64)
    ds = ArrayDataset((x, y))
    model = Sequential(layer_defs=[
        F("Flatten"),
        F("Dense", 2),
    ])
    mapped = Map(ds, Project(Pipe(Select(0), model), Select(1)))

    out = list(mapped)

    assert mapped.spec[0] == TensorSpec("float32", shape=(2,), backend="tf")
    assert [tuple(latent.shape) for latent, _ in out] == [(2,), (2,)]
    assert [int(label) for _, label in out] == [3, 7]


def test_tf_argmax_pipeline_after_model_output():
    from dryml.models.tf import Sequential

    x = np.zeros((2, 28, 28, 1), dtype=np.float32)
    y = np.array([3, 7], dtype=np.int64)
    ds = ArrayDataset((x, y))
    encoder = Sequential(layer_defs=[
        F("Flatten"),
        F("Dense", 2),
    ])
    classifier = Sequential(layer_defs=[
        F("Dense", 10),
    ])
    mapped = Map(ds, Project(Pipe(Select(0), encoder, classifier, ArgMax()), Select(1)))

    out = list(mapped)

    assert mapped.spec[0] == TensorSpec("int64", shape=(), backend="tf")
    assert [tuple(pred.shape) for pred, _ in out] == [(), ()]
    assert [int(label) for _, label in out] == [3, 7]


def test_tf_autoencoder_map_unbatched_image_uses_child_model_bindings():
    from dryml.models.tf import Sequential

    x = np.zeros((2, 28, 28, 1), dtype=np.float32)
    ds = ArrayDataset(x)
    encoder = Sequential(layer_defs=[
        F("Flatten"),
        F("Dense", 2),
    ])
    decoder = Sequential(layer_defs=[
        F("Dense", 28 * 28),
        F("Reshape", (28, 28, 1)),
    ])
    model = AutoEncoder(encoder=encoder, decoder=decoder)
    mapped = Map(ds, model)

    out = list(mapped)

    assert mapped.spec == TensorSpec("float32", shape=(28, 28, 1), backend="tf")
    assert [tuple(item.shape) for item in out] == [(28, 28, 1), (28, 28, 1)]


def test_tf_model_spec_inference_rejects_dataset_element_tuple():
    from dryml.models.tf import Sequential

    x = np.zeros((4, 3), dtype=np.float32)
    y = np.zeros((4,), dtype=np.int64)
    ds = ArrayDataset((x, y))
    model = Sequential(layer_defs=(F("Dense", units=2),))

    with pytest.raises(ValueError, match="Input spec structure does not match"):
        Map(ds, model)

    assert Map(ds, Select(0), model).spec == TensorSpec("float32", shape=(2,), backend="tf")


def _autoencoder_data():
    x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return ArrayDataset((x, x.copy()))


def _autoencoder_model():
    from dryml.models.tf import Sequential

    encoder = Sequential(
        layer_defs=(
            F("Dense", units=8, activation="relu"),
            F("Dense", units=2, activation="linear"),
        )
    )
    decoder = Sequential(
        layer_defs=(
            F("Dense", units=8, activation="relu"),
            F("Dense", units=3, activation="linear"),
        )
    )
    return AutoEncoder(encoder=encoder, decoder=decoder)


def test_tf_basic_training_builds_keras_adapter_for_autoencoder():
    from dryml.models.tf import BasicTraining, Wrapper

    model = _autoencoder_model()
    train_fn = BasicTraining(epochs=1, batch_size=2, verbose=0)
    exp = Experiment(
        model,
        train_fn,
        train_data=_autoencoder_data(),
        optimizer=Wrapper(tf.keras.optimizers.SGD, learning_rate=0.01),
        loss=Wrapper(tf.keras.losses.MeanSquaredError),
    )

    history = train_fn(exp)

    assert history is not None
    assert exp.state.epoch == 1
    assert exp.state.step == 2
    assert exp.state.examples_seen == 4
    assert model.encoder.obj.trainable_variables
    assert model.decoder.obj.trainable_variables


def test_tf_basic_training_repeats_finite_dataset_for_multiple_epochs():
    from dryml.models.tf import BasicTraining, Wrapper

    model = _autoencoder_model()
    train_fn = BasicTraining(epochs=2, batch_size=2, verbose=0)
    exp = Experiment(
        model,
        train_fn,
        train_data=_autoencoder_data(),
        optimizer=Wrapper(tf.keras.optimizers.SGD, learning_rate=0.01),
        loss=Wrapper(tf.keras.losses.MeanSquaredError),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        history = train_fn(exp)

    assert len(history.epoch) == 2
    assert exp.state.epoch == 2
    assert exp.state.step == 4
    assert not any("Your input ran out of data" in str(warning.message) for warning in caught)


def test_tf_training_gradient_tape_trains_autoencoder():
    from dryml.core.repo import get_default_repo
    from dryml.models.tf import Loss, Optimizer, Training

    assert get_default_repo() is None
    model = _autoencoder_model()
    train_fn = Training(
        optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.01),
        loss=Loss(tf.keras.losses.MeanSquaredError),
        epochs=1,
        batch_size=2,
        verbose=0,
    )
    exp = Experiment(model, train_fn, train_data=_autoencoder_data())
    observed = []

    losses = train_fn(exp, callbacks=(lambda: observed.append(exp.state.examples_seen),))

    assert len(losses) == 2
    assert exp.state.epoch == 1
    assert exp.state.step == 2
    assert exp.state.examples_seen == 4
    assert observed == [2, 4]
    assert model.encoder.obj.trainable_variables
    assert model.decoder.obj.trainable_variables
    assert get_default_repo() is None


def test_tf_keras_callbacks_observe_post_update_retained_accounting():
    from dryml.models.tf import BasicTraining, Loss, Model, Optimizer

    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = x * 2.0
    exp = Experiment(
        Model(TinyKerasModel),
        BasicTraining(
            optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.01),
            loss=Loss(tf.keras.losses.MeanSquaredError),
            epochs=1,
            batch_size=2,
            verbose=0,
        ),
        train_data=ArrayDataset((x, y)),
    )
    observed = []

    exp.train_fn(exp, callbacks=(lambda: observed.append((exp.state.step, exp.state.examples_seen)),))

    assert observed == [(1, 2), (2, 4)]


def _tf_accounting_experiment(*, repo=None, count=64, batch_size=1, trainer=None):
    from dryml.models.tf import Loss, Model, Optimizer, Training

    tf.keras.utils.set_random_seed(7103)
    x = np.arange(count, dtype=np.float32).reshape(-1, 1) / 10.0
    y = x * 2.0
    model = Model(TinyKerasModel, repo=repo)
    return Experiment(
        model,
        trainer or Training(
            optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.01, repo=repo),
            loss=Loss(tf.keras.losses.MeanSquaredError, repo=repo),
            epochs=1,
            batch_size=batch_size,
            verbose=0,
            repo=repo,
        ),
        train_data=ArrayDataset((x, y), repo=repo),
        repo=repo,
    )


def test_tf_interrupt_at_step_32_resumes_exact_remaining_epoch(tmp_path):
    repo = Repo(stores=tmp_path)
    interrupted = _tf_accounting_experiment(repo=repo)
    seen = []

    def interrupt_at_32():
        seen.append((interrupted.state.step, interrupted.state.epoch, interrupted.state.next_batch))
        if interrupted.state.step == 32:
            raise RuntimeError("pause at retained step 32")

    with pytest.raises(RuntimeError, match="step 32"):
        interrupted.train_fn(interrupted, callbacks=(interrupt_at_32,))
    checkpoint = repo.save_object(interrupted, deep_capture=True)
    resumed = Repo(stores=tmp_path).load_state_ref(checkpoint, reuse_live="never")
    remaining_losses = resumed.train_fn(resumed)

    baseline = _tf_accounting_experiment()
    baseline_losses = baseline.train_fn(baseline)

    assert seen == [(step, 0, step) for step in range(1, 33)]
    assert len(remaining_losses) == 32
    assert len(baseline_losses) == 64
    assert resumed.state.step == baseline.state.step == 64
    assert resumed.state.examples_seen == baseline.state.examples_seen == 64
    assert resumed.state.epoch == baseline.state.epoch == 1
    assert resumed.state.next_batch == baseline.state.next_batch == 0
    assert resumed.state.loss_denominator == baseline.state.loss_denominator == 64
    for actual, expected in zip(resumed.model.obj.variables, baseline.model.obj.variables):
        np.testing.assert_allclose(actual.numpy(), expected.numpy())
    assert int(resumed.train_fn.optimizer.obj.iterations.numpy()) == int(
        baseline.train_fn.optimizer.obj.iterations.numpy()
    )


def test_tf_trainers_accept_a_legacy_exhausted_saved_epoch_without_empty_data_error():
    from dryml.models.tf import BasicTraining, Loss, Model, Optimizer, Training

    data = ArrayDataset((np.zeros((2, 1), dtype=np.float32), np.zeros((2, 1), dtype=np.float32)))
    for trainer_type in (BasicTraining, Training):
        exp = Experiment(
            Model(TinyKerasModel),
            trainer_type(
                optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.0),
                loss=Loss(tf.keras.losses.MeanSquaredError),
                epochs=1,
                batch_size=1,
                verbose=0,
            ),
            train_data=data,
        )
        exp.state.next_batch = 2
        exp.state.target_epoch = 1

        result = exp.train_fn(exp)

        assert exp.state.epoch == 1
        assert exp.state.next_batch == 0
        assert exp.state.target_epoch is None
        if trainer_type is Training:
            assert result == []


def test_keras_resume_later_epoch_and_short_final_batch_keep_target_and_actual_accounting(tmp_path):
    from dryml.models.tf import BasicTraining, Loss, Model, Optimizer

    repo = Repo(stores=tmp_path)
    tf.keras.utils.set_random_seed(7103)
    x = np.zeros((5, 1), dtype=np.float32)
    y = np.concatenate((np.ones((4, 1), dtype=np.float32), np.full((1, 1), 5.0, dtype=np.float32)))
    train_fn = BasicTraining(
        optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.0, repo=repo),
        loss=Loss(tf.keras.losses.MeanSquaredError, repo=repo),
        epochs=2,
        batch_size=2,
        verbose=0,
        repo=repo,
    )
    interrupted = Experiment(Model(ZeroKerasModel, repo=repo), train_fn, train_data=ArrayDataset((x, y), repo=repo), repo=repo)
    observed = []

    def interrupt_in_second_epoch():
        observed.append((interrupted.state.step, interrupted.state.epoch, interrupted.state.next_batch))
        if interrupted.state.step == 4:
            raise RuntimeError("pause in epoch two")

    with pytest.raises(RuntimeError, match="epoch two"):
        interrupted.train_fn(interrupted, callbacks=(interrupt_in_second_epoch,))
    checkpoint = repo.save_object(interrupted, deep_capture=True)
    resumed = Repo(stores=tmp_path).load_state_ref(checkpoint, reuse_live="never")
    resumed_updates = []
    resumed.train_fn(
        resumed,
        callbacks=(lambda: resumed_updates.append((resumed.state.step, resumed.state.epoch, resumed.state.next_batch)),),
    )

    assert observed[-1] == (4, 1, 1)
    assert resumed_updates == [(5, 1, 2), (6, 2, 0)]
    assert resumed.state.step == 6
    assert resumed.state.examples_seen == 10
    assert resumed.state.epoch == 2
    assert resumed.state.next_batch == 0
    assert resumed.state.target_epoch is None

    short = Experiment(
        Model(ZeroKerasModel),
        BasicTraining(
            optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.0),
            loss=Loss(tf.keras.losses.MeanSquaredError),
            epochs=1,
            batch_size=2,
            verbose=0,
        ),
        train_data=ArrayDataset((x, y)),
    )
    final_positions = []
    short.train_fn(short, callbacks=(lambda: final_positions.append((short.state.epoch, short.state.next_batch)),))

    assert short.state.examples_seen == 5
    assert short.state.loss_denominator == 5
    assert short.state.loss_numerator / short.state.loss_denominator == pytest.approx(29 / 5)
    assert final_positions == [(0, 1), (0, 2), (1, 0)]


def test_keras_preflights_callbacks_and_mean_loss_before_model_or_optimizer_mutation():
    from dryml.models.tf import BasicTraining, Loss, Model, Optimizer, Training

    model = Model(TinyKerasModel)
    optimizer = Optimizer(tf.keras.optimizers.SGD, learning_rate=0.01)
    exp = Experiment(
        model,
        BasicTraining(optimizer=optimizer, loss=Loss(tf.keras.losses.MeanSquaredError), batch_size=1, verbose=0),
        train_data=ArrayDataset((np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32))),
    )
    optimizer_values = [value.numpy().copy() for value in optimizer.obj.variables]

    with pytest.raises(TypeError, match="callbacks"):
        exp.train_fn(exp, callbacks=(lambda: None, object()))
    assert model.obj.built is False
    assert exp.state.step == exp.state.examples_seen == 0
    for actual, expected in zip(optimizer.obj.variables, optimizer_values):
        np.testing.assert_allclose(actual.numpy(), expected)

    rejecting = BasicTraining(
        optimizer=optimizer,
        loss=Loss(tf.keras.losses.MeanSquaredError, reduction="sum"),
        batch_size=1,
        verbose=0,
    )
    with pytest.raises(ValueError, match="mean reduction"):
        rejecting(exp)
    assert model.obj.built is False
    assert exp.state.step == exp.state.examples_seen == 0

    rejecting_loop = Training(
        optimizer=optimizer,
        loss=Loss(tf.keras.losses.MeanSquaredError, reduction="sum"),
        batch_size=1,
        verbose=0,
    )
    with pytest.raises(ValueError, match="mean reduction"):
        rejecting_loop(exp)
    assert model.obj.built is False
    assert exp.state.step == exp.state.examples_seen == 0


def test_keras_internal_accounting_precedes_native_callbacks_and_uses_owned_train_step_loss():
    from dryml.models.tf import BasicTraining, Loss, Model, Optimizer

    events = []

    class NativeCallback(tf.keras.callbacks.Callback):
        def __init__(self, state, sink):
            super().__init__()
            self.state = state
            self.sink = sink

        def on_train_batch_end(self, batch, logs=None):
            del batch, logs
            self.sink.append(("native", self.state.step, self.state.examples_seen))

    x = np.zeros((3, 1), dtype=np.float32)
    y = np.array([[1.0], [3.0], [5.0]], dtype=np.float32)
    train_fn = BasicTraining(
        optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.0),
        loss=Loss(tf.keras.losses.MeanSquaredError),
        batch_size=2,
        verbose=0,
    )
    exp = Experiment(
        Model(ZeroKerasModel),
        train_fn,
        train_data=ArrayDataset((x, y)),
    )
    train_fn.callbacks = (NativeCallback(exp.state, events),)

    exp.train_fn(exp)

    assert events == [
        ("native", 1, 2),
        ("native", 2, 3),
    ]
    assert exp.state.loss_numerator / exp.state.loss_denominator == pytest.approx(35 / 3)


def test_tf_training_prepares_cross_backend_data_before_model_invocation(monkeypatch):
    from dryml.models.tf import Loss, Model, Optimizer, Training
    from dryml.models.utils import TrainingPreparation

    events = []
    original_prepare = TrainingPreparation.prepare

    def prepared(self, *args, **kwargs):
        events.append("prepare")
        return original_prepare(self, *args, **kwargs)

    monkeypatch.setattr(TrainingPreparation, "prepare", prepared)
    model = Model(TinyKerasModel)
    original_call = model.obj.call

    def record_call(value):
        events.append("model")
        return original_call(value)

    monkeypatch.setattr(model.obj, "call", record_call)
    exp = Experiment(
        model,
        Training(
            optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.0),
            loss=Loss(tf.keras.losses.MeanSquaredError),
            epochs=1,
            batch_size=1,
            verbose=0,
        ),
        train_data=ArrayDataset((np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32))),
    )

    exp.train_fn(exp)

    assert events[:2] == ["prepare", "model"]


def test_tf_final_callback_resume_runs_one_validation_postlude_without_an_update(monkeypatch):
    from dryml.models.tf import Loss, Model, Optimizer, Training
    import dryml.models.tf.base as tf_base

    postludes = []

    class Progress:
        def __init__(self, **kwargs):
            del kwargs

        def update(self, *args, **kwargs):
            del args, kwargs

        def epoch_end(self, *args, **kwargs):
            postludes.append("progress")

        def close(self):
            pass

    monkeypatch.setattr(tf_base, "TrainingProgress", Progress)
    trainer = Training(
        optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.0),
        loss=Loss(tf.keras.losses.MeanSquaredError), epochs=1, batch_size=1, verbose=0,
    )
    data = ArrayDataset((np.zeros((2, 1), dtype=np.float32), np.zeros((2, 1), dtype=np.float32)))
    exp = Experiment(Model(TinyKerasModel), trainer, train_data=data, val_data=data)
    evaluations = []
    monkeypatch.setattr(trainer, "_evaluate", lambda *args: evaluations.append("validation") or {})

    def interrupt_final_callback():
        if exp.state.step == 2:
            assert (exp.state.epoch, exp.state.next_batch) == (1, 0)
            raise RuntimeError("final callback")

    with pytest.raises(RuntimeError, match="final callback"):
        trainer(exp, callbacks=(interrupt_final_callback,))

    assert exp.state.pending_epoch_postlude == 0
    assert exp.state.step == 2
    trainer(exp)
    assert exp.state.target_epoch is None
    assert exp.state.pending_epoch_postlude is None
    assert exp.state.step == 2
    assert evaluations == ["validation"]
    assert postludes == ["progress"]

    trainer(exp)
    assert exp.state.step == 4


def test_keras_rejects_unreplayable_native_callback_safe_point_before_training():
    from dryml.models.tf import BasicTraining, Loss, Model, Optimizer

    events = []

    class Native(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            del logs
            events.append(("batch", batch))

        def on_epoch_end(self, epoch, logs=None):
            del logs
            events.append(("epoch", epoch))

    trainer = BasicTraining(
        optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.0),
        loss=Loss(tf.keras.losses.MeanSquaredError), epochs=1, batch_size=1,
        verbose=0,
    )
    trainer.callbacks = (Native(),)
    data = ArrayDataset((np.zeros((2, 1), dtype=np.float32), np.zeros((2, 1), dtype=np.float32)))
    exp = Experiment(Model(TinyKerasModel), trainer, train_data=data)

    with pytest.raises(ValueError, match="does not support native callbacks"):
        trainer(exp, callbacks=(lambda: None,))

    assert events == []
    assert int(trainer.optimizer.obj.iterations.numpy()) == 0
    assert exp.state.target_epoch is None


def test_keras_rejects_untruthful_objective_weights_before_training_mutation():
    from dryml.models.tf import BasicTraining, Loss, Model, Optimizer

    model = Model(TinyKerasModel)
    exp = Experiment(
        model,
        BasicTraining(
            optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.0),
            loss=Loss(tf.keras.losses.MeanSquaredError), batch_size=1, verbose=0,
            fit_kwargs={"class_weight": {0: 1.0}},
        ),
        train_data=ArrayDataset((np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32))),
    )

    with pytest.raises(ValueError, match="class_weight"):
        exp.train_fn(exp)

    assert model.obj.built is False
    assert exp.state.target_epoch is None

def test_keras_retains_exact_static_and_dynamic_regularized_objectives():
    from dryml.models.tf import BasicTraining, Loss, Model, Optimizer

    class StaticRegularizer(tf.keras.Sequential):
        def __init__(self):
            super().__init__([
                tf.keras.layers.Dense(
                    1,
                    use_bias=False,
                    kernel_initializer="ones",
                    kernel_regularizer=tf.keras.regularizers.L2(0.5),
                ),
            ])

    def run(model_type, x, y, expected_loss):
        model = Model(model_type)
        optimizer = Optimizer(tf.keras.optimizers.SGD, learning_rate=0.0)
        exp = Experiment(
            model,
            BasicTraining(
                optimizer=optimizer,
                loss=Loss(tf.keras.losses.MeanSquaredError),
                batch_size=1,
                verbose=0,
            ),
            train_data=ArrayDataset((x, y)),
        )

        exp.train_fn(exp)

        assert exp.state.loss_numerator / exp.state.loss_denominator == pytest.approx(expected_loss)
        assert (exp.state.step, exp.state.examples_seen) == (1, 1)
        assert int(optimizer.obj.iterations.numpy()) == 1
        return model, exp

    run(
        StaticRegularizer,
        np.array([[2.0]], dtype=np.float32),
        np.array([[0.0]], dtype=np.float32),
        4.5,
    )
    dynamic_model, dynamic_exp = run(
        NestedDynamicKerasModel,
        np.array([[2.0]], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        1.25,
    )
    assert dynamic_model.obj.regularizer.normalizer.moving_mean.numpy()[0] != 0.0
    assert dynamic_exp.state.phase is None


def test_keras_rejects_grouped_updates_before_mutation(monkeypatch):
    from dryml.models.tf import BasicTraining, Loss, Model, Optimizer

    data = ArrayDataset((np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)))
    model = Model(TinyKerasModel)
    optimizer = Optimizer(tf.keras.optimizers.SGD, learning_rate=0.0)
    exp = Experiment(
        model,
        BasicTraining(
            optimizer=optimizer, loss=Loss(tf.keras.losses.MeanSquaredError),
            compile_kwargs={"steps_per_execution": 2}, batch_size=1, verbose=0,
        ),
        train_data=data,
    )
    with pytest.raises(ValueError, match="steps_per_execution"):
        exp.train_fn(exp)
    assert model.obj.built is False
    assert int(optimizer.obj.iterations.numpy()) == 0
    assert exp.state.target_epoch is None

    accumulation_optimizer = Optimizer(tf.keras.optimizers.SGD, learning_rate=0.0)
    monkeypatch.setattr(accumulation_optimizer.obj, "gradient_accumulation_steps", 2, raising=False)
    accumulating = Experiment(
        Model(TinyKerasModel),
        BasicTraining(
            optimizer=accumulation_optimizer, loss=Loss(tf.keras.losses.MeanSquaredError), batch_size=1, verbose=0,
        ),
        train_data=data,
    )
    with pytest.raises(ValueError, match="gradient accumulation"):
        accumulating.train_fn(accumulating)
    assert int(accumulation_optimizer.obj.iterations.numpy()) == 0
    assert accumulating.state.target_epoch is None

def test_keras_rejects_dryml_and_native_callback_recovery_before_setup():
    from dryml.models.tf import BasicTraining, Loss, Model, Optimizer

    events = []

    class Native(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            del logs
            events.append("native")

    model = Model(TinyKerasModel)
    optimizer = Optimizer(tf.keras.optimizers.SGD, learning_rate=0.0)
    model._pending_restore_path = "/invalid/model/checkpoint"
    optimizer._pending_restore_path = "/invalid/optimizer/checkpoint"
    trainer = BasicTraining(
        optimizer=optimizer,
        loss=Loss(tf.keras.losses.MeanSquaredError),
        batch_size=1,
        verbose=0,
    )
    trainer.fit_kwargs["callbacks"] = [Native()]
    exp = Experiment(
        model,
        trainer,
        train_data=ArrayDataset((np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32))),
    )

    with pytest.raises(ValueError, match="does not support native callbacks"):
        trainer(exp, callbacks=(lambda: None,))

    assert events == []
    assert model._pending_restore_path == "/invalid/model/checkpoint"
    assert optimizer._pending_restore_path == "/invalid/optimizer/checkpoint"
    assert model.obj.built is False
    assert int(optimizer.obj.iterations.numpy()) == 0
    assert trainer.method_graph().conversion_edges == ()
    assert exp.state.phase == TrainState.initial


def test_training_preparation_resets_current_invocation_graph_facts():
    from dryml.models.tf import BasicTraining, Loss, Model, Optimizer
    from dryml.models.utils import TrainingPreparation

    trainer = BasicTraining(
        optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.0),
        loss=Loss(tf.keras.losses.MeanSquaredError),
        batch_size=1,
        verbose=0,
    )
    first = ArrayDataset((np.zeros((2, 1), dtype=np.float32), np.zeros((2, 1), dtype=np.float32)))
    second = ArrayDataset((np.zeros((2, 2), dtype=np.float32), np.zeros((2, 1), dtype=np.float32)))
    first_xy = trainer._xy_data(trainer._prepare_data(first, for_training=True))
    second_xy = trainer._xy_data(trainer._prepare_data(second, for_training=True))

    trainer._begin_training_preparation_generation()
    training = TrainingPreparation.from_specs(trainer, first_xy.spec[0], first_xy.spec[1], "tf")
    validation = TrainingPreparation.from_specs(trainer, second_xy.spec[0], second_xy.spec[1], "tf")
    graph = trainer.method_graph()
    assert graph.input_specs == (*training.producer_specs, *validation.producer_specs)

    trainer._begin_training_preparation_generation()
    current = TrainingPreparation.from_specs(trainer, second_xy.spec[0], second_xy.spec[1], "tf")
    assert graph.input_specs == current.producer_specs
    assert len(graph.source_nodes) == 1

    exp = Experiment(Model(TinyKerasModel), trainer, train_data=first, val_data=first)
    trainer(exp)
    assert len(graph.source_nodes) == 2
    exp.val_data = None
    trainer(exp)
    assert graph.input_specs == trainer.training_preparation.producer_specs
    assert len(graph.source_nodes) == 1


def test_keras_zero_epoch_does_not_emit_native_lifecycle_callbacks():
    from dryml.models.tf import BasicTraining, Loss, Model, Optimizer

    events = []

    class Native(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            events.append("begin")

        def on_train_end(self, logs=None):
            events.append("end")

    trainer = BasicTraining(
        optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.0),
        loss=Loss(tf.keras.losses.MeanSquaredError), epochs=0, batch_size=1, verbose=0,
    )
    trainer.callbacks = (Native(),)
    exp = Experiment(Model(TinyKerasModel), trainer, train_data=ArrayDataset((np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32))))

    exp.train_fn(exp)

    assert events == []
    assert exp.state.target_epoch is None


def test_tf_setup_failure_and_validation_failure_leave_only_recoverable_targets(monkeypatch):
    from dryml.models.tf import Loss, Model, Optimizer, Training

    trainer = Training(
        optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.0),
        loss=Loss(tf.keras.losses.MeanSquaredError), epochs=1, batch_size=1, verbose=0,
    )
    exp = Experiment(Model(TinyKerasModel), trainer)
    with pytest.raises(ValueError, match="train_data"):
        exp.train_fn(exp)
    assert exp.state.target_epoch is None

    data = ArrayDataset((np.zeros((2, 1), dtype=np.float32), np.zeros((2, 1), dtype=np.float32)))
    exp.train_data = data
    exp.val_data = data
    original_evaluate = trainer._evaluate
    calls = []

    def fail_once(*args):
        calls.append("validation")
        if len(calls) == 1:
            raise RuntimeError("validation failed")
        return original_evaluate(*args)

    monkeypatch.setattr(trainer, "_evaluate", fail_once)
    with pytest.raises(RuntimeError, match="validation failed"):
        exp.train_fn(exp)
    assert (exp.state.epoch, exp.state.next_batch, exp.state.pending_epoch_postlude) == (1, 0, 0)
    assert exp.state.target_epoch == 1
    assert exp.state.step == 2

    exp.train_fn(exp)
    assert exp.state.target_epoch is None
    assert exp.state.step == 2
    exp.train_fn(exp)
    assert exp.state.step == 4


def test_keras_early_stopping_completes_a_shortened_target():
    from dryml.models.tf import BasicEarlyStoppingTraining, Loss, Model, Optimizer

    data = ArrayDataset((np.zeros((2, 1), dtype=np.float32), np.zeros((2, 1), dtype=np.float32)))
    exp = Experiment(
        Model(ZeroKerasModel),
        BasicEarlyStoppingTraining(
            optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.0),
            loss=Loss(tf.keras.losses.MeanSquaredError), epochs=5, batch_size=1,
            patience=0, monitor="loss", verbose=0,
        ),
        train_data=data,
    )

    exp.train_fn(exp)

    assert 0 < exp.state.epoch < 5
    assert exp.state.target_epoch is None
    previous_steps = exp.state.step
    exp.train_fn(exp)
    assert exp.state.step > previous_steps


def test_keras_unknown_finite_stream_completes_without_callbacks_and_rejects_safe_points():
    from dryml.models.tf import BasicTraining, Loss, Model, Optimizer

    class UnknownArrayDataset(ArrayDataset):
        def __len__(self):
            raise NotImplementedError

    data = UnknownArrayDataset((np.zeros((2, 1), dtype=np.float32), np.zeros((2, 1), dtype=np.float32)))
    trainer = BasicTraining(
        optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.0),
        loss=Loss(tf.keras.losses.MeanSquaredError), epochs=2, batch_size=1, verbose=0,
    )
    exp = Experiment(Model(TinyKerasModel), trainer, train_data=data)

    trainer(exp)
    assert (exp.state.epoch, exp.state.next_batch, exp.state.target_epoch) == (2, 0, None)
    assert exp.state.step == 4

    blocked = Experiment(Model(TinyKerasModel), trainer, train_data=data)
    with pytest.raises(ValueError, match="finite deterministic"):
        trainer(blocked, callbacks=(lambda: None,))
    assert blocked.state.target_epoch is None


def test_keras_unknown_stream_retains_validation_postlude_before_failure():
    from dryml.models.tf import BasicTraining, Loss, Model, Optimizer

    class UnknownArrayDataset(ArrayDataset):
        def __len__(self):
            raise NotImplementedError

    class ValidationFailsOnce(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense = tf.keras.layers.Dense(1)
            self.failed = False

        def call(self, value, training=None):
            if training is False and not self.failed:
                self.failed = True
                raise RuntimeError("validation postlude")
            return self.dense(value)

    data = UnknownArrayDataset((np.zeros((2, 1), dtype=np.float32), np.zeros((2, 1), dtype=np.float32)))
    optimizer = Optimizer(tf.keras.optimizers.SGD, learning_rate=0.0)
    exp = Experiment(
        Model(ValidationFailsOnce),
        BasicTraining(
            optimizer=optimizer, loss=Loss(tf.keras.losses.MeanSquaredError), epochs=1, batch_size=1, verbose=0,
        ),
        train_data=data,
        val_data=data,
    )

    with pytest.raises(RuntimeError, match="validation postlude"):
        exp.train_fn(exp)
    assert (exp.state.epoch, exp.state.next_batch, exp.state.pending_epoch_postlude) == (1, 0, 0)
    assert exp.state.pending_epoch_postlude_phase == "start"
    iterations = int(optimizer.obj.iterations.numpy())
    assert iterations > 0

    exp.train_fn(exp)
    assert exp.state.target_epoch is None
    assert exp.state.pending_epoch_postlude is None
    assert int(optimizer.obj.iterations.numpy()) == iterations


def test_keras_resume_skips_absent_validation_for_retained_validation_phase():
    from types import SimpleNamespace

    from dryml.models import TrainState
    from dryml.models.tf.base import _resume_keras_postlude

    class TrainingModel:
        def evaluate(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("validation must not run without validation data")

    state = TrainState(
        epoch=1,
        target_epoch=1,
        pending_epoch_postlude=0,
        pending_epoch_postlude_phase="validation",
    )

    _resume_keras_postlude(
        SimpleNamespace(state=state),
        TrainingModel(),
        (),
        validation_data=None,
        validation_steps=None,
        steps_per_epoch=1,
    )

    assert state.pending_epoch_postlude is None
    assert state.pending_epoch_postlude_phase is None


def test_tf_unknown_stream_retains_validation_metrics_across_progress_retry(monkeypatch):
    from dryml.models.tf import Loss, Model, Optimizer, Training
    import dryml.models.tf.base as tf_base

    class UnknownArrayDataset(ArrayDataset):
        def __len__(self):
            raise NotImplementedError

    class Progress:
        attempts = 0
        received = []

        def __init__(self, **kwargs):
            del kwargs

        def update(self, *args, **kwargs):
            del args, kwargs

        def epoch_end(self, *args, **kwargs):
            type(self).attempts += 1
            type(self).received.append(kwargs["metrics"])
            if type(self).attempts == 1:
                raise RuntimeError("progress postlude")

        def close(self):
            pass

    monkeypatch.setattr(tf_base, "TrainingProgress", Progress)
    trainer = Training(
        optimizer=Optimizer(tf.keras.optimizers.SGD, learning_rate=0.0),
        loss=Loss(tf.keras.losses.MeanSquaredError), epochs=1, batch_size=1, verbose=0,
    )
    data = UnknownArrayDataset((np.zeros((2, 1), dtype=np.float32), np.zeros((2, 1), dtype=np.float32)))
    exp = Experiment(Model(TinyKerasModel), trainer, train_data=data, val_data=data)
    evaluations = []
    monkeypatch.setattr(trainer, "_evaluate", lambda *args: evaluations.append("validation") or {"loss": 3.0})

    with pytest.raises(RuntimeError, match="progress postlude"):
        exp.train_fn(exp)
    assert evaluations == ["validation"]
    retained_metrics = exp.state.pending_epoch_metrics
    assert retained_metrics is not None
    assert retained_metrics["val_loss"] == 3.0

    exp.train_fn(exp)
    assert evaluations == ["validation"]
    assert Progress.received == [retained_metrics, retained_metrics]
