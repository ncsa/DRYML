import numpy as np
import pytest

from dryml.data import ArrayDataset
from dryml.models import AutoEncoder, Experiment


torch = pytest.importorskip("torch")


def test_torch_basic_training_updates_experiment_state():
    from dryml.models.torch import BasicTraining, Model, Optimizer

    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32)
    ds = ArrayDataset((x, y))

    model = Model(torch.nn.Linear, 1, 1)
    optimizer = Optimizer(torch.optim.SGD, target=model, lr=0.01)
    train_fn = BasicTraining(
        optimizer=optimizer,
        loss_cls=torch.nn.MSELoss,
        epochs=2,
        batch_size=2,
    )
    exp = Experiment(model, train_fn, train_data=ds)

    losses = exp.train()

    assert len(losses) == 4
    assert exp.state.epoch == 2
    assert exp.state.step == 4
    assert exp.state.phase == "trained"
    assert optimizer.obj is not None
    assert optimizer.obj.param_groups[0]["lr"] == 0.01


def test_torch_autoencoder_optimizer_targets_composite_model():
    from dryml.models.torch import BasicTraining, Optimizer, Sequential

    x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    ds = ArrayDataset((x, x.copy()))
    encoder = Sequential(
        layer_defs=(
            ("Linear", (3, 8), {}),
            ("ReLU", {}),
            ("Linear", (8, 2), {}),
        )
    )
    decoder = Sequential(
        layer_defs=(
            ("Linear", (2, 8), {}),
            ("ReLU", {}),
            ("Linear", (8, 3), {}),
        )
    )
    model = AutoEncoder(encoder=encoder, decoder=decoder)
    optimizer = Optimizer(torch.optim.SGD, target=model, lr=0.05)
    train_fn = BasicTraining(
        optimizer=optimizer,
        loss_cls=torch.nn.MSELoss,
        epochs=2,
        batch_size=2,
    )
    exp = Experiment(model, train_fn, train_data=ds)

    expected_params = sum(1 for _ in encoder.trainable_parameters("torch")) + sum(
        1 for _ in decoder.trainable_parameters("torch")
    )
    actual_params = sum(len(group["params"]) for group in optimizer.obj.param_groups)
    losses = exp.train()

    assert actual_params == expected_params
    assert len(losses) == 4
    assert exp.state.phase == "trained"
