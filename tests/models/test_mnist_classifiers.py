import pytest

from dryml.core.tensor_spec import TensorSpec
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
