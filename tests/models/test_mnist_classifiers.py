import pytest

import dryml.execute as execute
from dryml.core2 import Repo, definition_mode
from dryml.core2.tensor_spec import TensorSpec
from dryml.data import Cast, Flatten, Map, Scale, Select, TFDSAdapter, Zip
from dryml.metrics import categorical_accuracy
from dryml.models import Experiment


pytest.importorskip("tensorflow_datasets")

_MNIST_TRAIN_SPLIT = "train[:64]"
_MNIST_VAL_SPLIT = "test[:32]"
_MNIST_EPOCHS = 1
_MNIST_SKLEARN_MAX_ITER = 2


def _mnist_dataset(split):
    raw = TFDSAdapter("mnist", split=split, as_supervised=True, as_numpy=True)
    x = Map(raw, Select(0), Cast("float32"), Scale.from_range(0.0, 255.0), Flatten())
    y = Map(raw, Select(1))
    return Zip(x, y)


def test_sklearn_basic_mnist_classifier_with_tfds_adapter():
    linear_model = pytest.importorskip("sklearn.linear_model")
    from dryml.models.sklearn import BasicTraining, ClassifierModel

    train_ds = _mnist_dataset(_MNIST_TRAIN_SPLIT)
    val_ds = _mnist_dataset(_MNIST_VAL_SPLIT)
    model = ClassifierModel(linear_model.SGDClassifier, loss="log_loss", max_iter=_MNIST_SKLEARN_MAX_ITER, tol=None, random_state=1)
    exp = Experiment(model, BasicTraining(), train_data=train_ds, val_data=val_ds)

    exp.train()

    assert categorical_accuracy(model, val_ds, batch_size=64) > 0.1
    assert Map(val_ds, Select(0), model).spec == TensorSpec("float32", shape=(10,), backend="numpy")
    assert exp.state.phase == "trained"


def test_tf_basic_mnist_classifier_with_tfds_adapter():
    tf = pytest.importorskip("tensorflow")
    from dryml.models.tf import BasicTraining, Loss, Optimizer, Sequential

    train_ds = _mnist_dataset(_MNIST_TRAIN_SPLIT)
    val_ds = _mnist_dataset(_MNIST_VAL_SPLIT)
    model = Sequential(
        layer_defs=(("Dense", {"units": 10}),),
    )
    optimizer = Optimizer(tf.keras.optimizers.SGD, learning_rate=0.5)
    train_fn = BasicTraining(
        optimizer=optimizer,
        loss=Loss(tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True),
        epochs=_MNIST_EPOCHS,
        batch_size=64,
    )
    exp = Experiment(model, train_fn, train_data=train_ds, val_data=val_ds)

    exp.train()

    assert categorical_accuracy(model, val_ds, batch_size=64) > 0.1
    assert Map(val_ds, Select(0), model).spec == TensorSpec("float32", shape=(10,), backend="tf")
    assert exp.state.phase == "trained"


def test_torch_basic_mnist_classifier_with_tfds_adapter():
    torch = pytest.importorskip("torch")
    from dryml.models.torch import Optimizer, Sequential, Training

    train_ds = _mnist_dataset(_MNIST_TRAIN_SPLIT)
    val_ds = _mnist_dataset(_MNIST_VAL_SPLIT)
    model = Sequential(
        layer_defs=(("Linear", (28 * 28, 10), {}),),
    )
    optimizer = Optimizer(torch.optim.SGD, target=model, lr=0.5)
    train_fn = Training(
        optimizer=optimizer,
        loss_cls=torch.nn.CrossEntropyLoss,
        epochs=_MNIST_EPOCHS,
        batch_size=64,
        verbose=0,
    )
    exp = Experiment(model, train_fn, train_data=train_ds, val_data=val_ds)

    exp.train()

    assert categorical_accuracy(model, val_ds, batch_size=64) > 0.1
    assert Map(val_ds, Select(0), model).spec == TensorSpec("float32", shape=(10,), backend="torch")
    assert exp.state.phase == "trained"


def _train_experiment_and_score(exp):
    exp.train()
    return categorical_accuracy(exp.model, exp.val_data, batch_size=64)


def _score_model(val_data, model):
    return categorical_accuracy(model, val_data, batch_size=64)


def sklearn_mnist_experiment():
    linear_model = pytest.importorskip("sklearn.linear_model")
    from dryml.models.sklearn import BasicTraining, ClassifierModel

    with definition_mode(concrete=True):
        train_ds = _mnist_dataset(_MNIST_TRAIN_SPLIT)
        val_ds = _mnist_dataset(_MNIST_VAL_SPLIT)
        model = ClassifierModel(
            linear_model.SGDClassifier,
            loss="log_loss",
            max_iter=_MNIST_SKLEARN_MAX_ITER,
            tol=None,
            random_state=1,
        )
        return Experiment(
            model,
            BasicTraining(),
            train_data=train_ds,
            val_data=val_ds,
        )


def tf_mnist_experiment():
    tf = pytest.importorskip("tensorflow")
    from dryml.models.tf import BasicTraining, Loss, Optimizer, Sequential

    with definition_mode(concrete=True):
        train_ds = _mnist_dataset(_MNIST_TRAIN_SPLIT)
        val_ds = _mnist_dataset(_MNIST_VAL_SPLIT)
        model = Sequential(
            layer_defs=(("Dense", {"units": 10}),),
        )
        optimizer = Optimizer(tf.keras.optimizers.SGD, learning_rate=0.5)
        train_fn = BasicTraining(
            optimizer=optimizer,
            loss=Loss(tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True),
            epochs=_MNIST_EPOCHS,
            batch_size=64,
        )
        return Experiment(model, train_fn, train_data=train_ds, val_data=val_ds)


def torch_mnist_experiment():
    torch = pytest.importorskip("torch")
    from dryml.models.torch import Optimizer, Sequential, Training

    with definition_mode(concrete=True):
        train_ds = _mnist_dataset(_MNIST_TRAIN_SPLIT)
        val_ds = _mnist_dataset(_MNIST_VAL_SPLIT)
        model = Sequential(
            layer_defs=(("Linear", (28 * 28, 10), {}),),
        )
        optimizer = Optimizer(torch.optim.SGD, target=model, lr=0.5)
        train_fn = Training(
            optimizer=optimizer,
            loss_cls=torch.nn.CrossEntropyLoss,
            epochs=_MNIST_EPOCHS,
            batch_size=64,
            verbose=0,
        )
        return Experiment(model, train_fn, train_data=train_ds, val_data=val_ds)


_MNIST_EXEC_CASES = [
    pytest.param(sklearn_mnist_experiment, "inline", id="sklearn-inline"),
    pytest.param(sklearn_mnist_experiment, "process", id="sklearn-process"),
    pytest.param(tf_mnist_experiment, "inline", id="tf-inline"),
    pytest.param(torch_mnist_experiment, "inline", id="torch-inline"),
]


@pytest.mark.parametrize(
    "experiment_factory,backend",
    _MNIST_EXEC_CASES,
)
def test_mnist_execute_multi_framework_train_and_score_matches_model_score(
        tmp_path,
        backend,
        experiment_factory):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo = Repo(stores=repo_dir)
    exp_def = experiment_factory()

    exp = repo.load_object(
        exp_def,
        restore_state=False,
        build_missing=True,
        cache="strong",
    )

    train_score = execute.run(
        _train_experiment_and_score,
        exp,
        backend=backend,
        repo=repo,
        update=[exp, exp.model],
    )
    model_score = execute.run(
        _score_model,
        exp.val_data,
        exp.model,
        backend=backend,
        repo=repo,
    )

    assert train_score == pytest.approx(model_score)
    assert train_score > 0.1
    assert exp.state.phase == "trained"


@pytest.mark.parametrize(
    "experiment_factory,backend",
    _MNIST_EXEC_CASES,
)
def test_mnist_execute_definition_only_train_and_score_matches_model_score(
        tmp_path,
        backend,
        experiment_factory):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo = Repo(stores=repo_dir)
    exp_def = experiment_factory()
    model_def = exp_def.parameters["model"]
    val_data_def = exp_def.parameters["val_data"]

    train_score = execute.run(
        _train_experiment_and_score,
        exp_def,
        backend=backend,
        repo=repo,
        update=[exp_def, model_def],
    )
    model_score = execute.run(
        _score_model,
        val_data_def,
        model_def,
        backend=backend,
        repo=repo,
    )

    assert train_score == pytest.approx(model_score)
    assert train_score > 0.1
