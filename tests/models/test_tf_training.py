import warnings

import numpy as np
import pytest

from dryml.core import FactorySpec, Repo
from dryml.core.tensor_spec import TensorSpec
from dryml.data import ArgMax, ArrayDataset, Map, Pipe, Project, Select
from dryml.models import AutoEncoder, Experiment


tf = pytest.importorskip("tensorflow")


class TinyKerasModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, x):
        return self.dense(x)


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

    history = exp.train()

    assert model.obj is model.mdl
    assert optimizer.obj is not None
    assert history is not None
    assert exp.state.epoch == 1
    assert exp.state.step == 2
    assert exp.state.phase == "trained"
    assert float(optimizer.obj.learning_rate.numpy()) == pytest.approx(0.01)


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
    exp.train()
    expected_predictions = model.obj(tf.convert_to_tensor(x)).numpy()
    expected_iterations = int(optimizer.obj.iterations.numpy())
    expected_optimizer = [value.numpy().copy() for value in optimizer.obj.variables]

    state = repo.save_object(exp, deep_capture=True)
    repo.close(flush=True)
    loaded = Repo(stores=tmp_path).load_state_ref(state, reuse_live="never")
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
    _, first = rebound.model.bind_first(
        tf.convert_to_tensor(x[0]), input_spec=ArrayDataset(x).spec
    )
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
    exp.train()
    state = repo.save_object(exp, deep_capture=True)

    exp.train()
    expected_predictions = model(tf.convert_to_tensor(x)).numpy()
    expected_optimizer = [value.numpy().copy() for value in optimizer.obj.variables]

    loaded = Repo(stores=tmp_path).load_state_ref(state, reuse_live="never")
    loaded.train()
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
    model = Sequential(layer_defs=(("Dense", {"units": 2}),))

    assert Map(ds, model).spec == TensorSpec("float32", shape=(2,), backend="tf")


def test_tf_sequential_accepts_factory_spec_and_constructor_tuple_shorthand():
    from dryml.models.tf import Sequential

    x = np.zeros((4, 32, 32, 1), dtype=np.float32)
    ds = ArrayDataset(x)
    encoder = Sequential(layer_defs=[
        ("Flatten",),
        ("Dense", 32, {"activation": "relu"}),
        FactorySpec("Dense", 2, activation="linear"),
    ])

    decoder = Sequential(layer_defs=[
        ("Dense", 32 * 32, {"activation": "linear"}),
        ("Reshape", ((32, 32, 1),), {}),
    ])

    assert Map(ds, encoder).spec == TensorSpec("float32", shape=(2,), backend="tf")
    assert decoder.infer_output_spec(TensorSpec("float32", shape=(2,), backend="tf")) == TensorSpec(
        "float32",
        shape=(32, 32, 1),
        backend="tf",
    )


def test_tf_model_map_unbatched_image_uses_backend_batch_axis():
    from dryml.models.tf import Sequential

    x = np.zeros((2, 28, 28, 1), dtype=np.float32)
    ds = ArrayDataset(x)
    model = Sequential(layer_defs=[
        ("Flatten",),
        ("Dense", 2),
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
        ("Flatten",),
        ("Dense", 2),
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
        ("Flatten",),
        ("Dense", 2),
    ])
    classifier = Sequential(layer_defs=[
        ("Dense", 10),
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
        ("Flatten",),
        ("Dense", 2),
    ])
    decoder = Sequential(layer_defs=[
        ("Dense", 28 * 28),
        ("Reshape", (28, 28, 1)),
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
    model = Sequential(layer_defs=(("Dense", {"units": 2}),))

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
            ("Dense", {"units": 8, "activation": "relu"}),
            ("Dense", {"units": 2, "activation": "linear"}),
        )
    )
    decoder = Sequential(
        layer_defs=(
            ("Dense", {"units": 8, "activation": "relu"}),
            ("Dense", {"units": 3, "activation": "linear"}),
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

    history = exp.train()

    assert history is not None
    assert exp.state.epoch == 1
    assert exp.state.step == 2
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
        history = exp.train()

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

    losses = exp.train()

    assert len(losses) == 2
    assert exp.state.epoch == 1
    assert exp.state.step == 2
    assert model.encoder.obj.trainable_variables
    assert model.decoder.obj.trainable_variables
    assert get_default_repo() is None
