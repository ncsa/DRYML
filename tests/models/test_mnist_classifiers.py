import pytest

from dryml.core2.tensor_spec import TensorSpec
from dryml.data import Map, Pack, Select, TFDSAdapter
from dryml.data.transforms import Cast, Flatten, Scale
from dryml.metrics import categorical_accuracy
from dryml.models import Experiment


pytest.importorskip("tensorflow_datasets")


def _mnist_dataset(split):
    raw = TFDSAdapter("mnist", split=split, as_supervised=True, as_numpy=True)
    x = Map(raw, Select(0), Cast("float32"), Scale.from_range(0.0, 255.0), Flatten())
    y = Map(raw, Select(1))
    return Pack(x, y)


def test_sklearn_basic_mnist_classifier_with_tfds_adapter():
    linear_model = pytest.importorskip("sklearn.linear_model")
    from dryml.models.sklearn import BasicTraining, ClassifierModel

    train_ds = _mnist_dataset("train[:1000]")
    val_ds = _mnist_dataset("test[:200]")
    model = ClassifierModel(linear_model.SGDClassifier, loss="log_loss", max_iter=20, tol=None, random_state=1)
    exp = Experiment(model, BasicTraining(), train_data=train_ds, val_data=val_ds)

    exp.train()

    assert categorical_accuracy(model, val_ds, batch_size=64) > 0.1
    assert exp.state.phase == "trained"


def test_tf_basic_mnist_classifier_with_tfds_adapter():
    tf = pytest.importorskip("tensorflow")
    from dryml.models.tf import BasicTraining, Loss, Optimizer
    from dryml.models.tf.keras import SequentialFunctionalModel

    train_ds = _mnist_dataset("train[:1000]")
    val_ds = _mnist_dataset("test[:200]")
    model = SequentialFunctionalModel(
        input_shape=(28 * 28,),
        layer_defs=(("Dense", {"units": 10}),),
        output_spec=TensorSpec("float32", shape=(10,), backend="tf"),
    )
    optimizer = Optimizer(tf.keras.optimizers.SGD, learning_rate=0.5)
    train_fn = BasicTraining(
        optimizer=optimizer,
        loss=Loss(tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True),
        epochs=5,
        batch_size=64,
    )
    exp = Experiment(model, train_fn, train_data=train_ds, val_data=val_ds)

    exp.train()

    assert categorical_accuracy(model, val_ds, batch_size=64) > 0.1
    assert exp.state.phase == "trained"


def test_torch_basic_mnist_classifier_with_tfds_adapter():
    torch = pytest.importorskip("torch")
    from dryml.models.torch import BasicTraining, Model, Optimizer

    train_ds = _mnist_dataset("train[:1000]")
    val_ds = _mnist_dataset("test[:200]")
    model = Model(
        torch.nn.Linear,
        28 * 28,
        10,
        device="cpu",
        output_spec=TensorSpec("float32", shape=(10,), backend="torch"),
    )
    optimizer = Optimizer(torch.optim.SGD, mdl=model, lr=0.5)
    train_fn = BasicTraining(
        optimizer=optimizer,
        loss_cls=torch.nn.CrossEntropyLoss,
        epochs=5,
        batch_size=64,
        device="cpu",
    )
    exp = Experiment(model, train_fn, train_data=train_ds, val_data=val_ds)

    exp.train()

    assert categorical_accuracy(model, val_ds, batch_size=64) > 0.1
    assert exp.state.phase == "trained"
