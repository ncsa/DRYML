import numpy as np
import pytest

from dryml.data import ArrayDataset
from dryml.models import Experiment


torch = pytest.importorskip("torch")


def test_torch_basic_training_updates_experiment_state():
    from dryml.models.torch import BasicTraining, Model, Optimizer

    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32)
    ds = ArrayDataset((x, y))

    model = Model(torch.nn.Linear, 1, 1)
    optimizer = Optimizer(torch.optim.SGD, mdl=model, lr=0.01)
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
