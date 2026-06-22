import numpy as np
import pytest
import sys

from dryml.core2.tensor_spec import TensorSpec
from dryml.data import ArgMax, ArrayDataset, Map, Pipe, Project, Select
from dryml.core2 import Repo
from dryml.models import AutoEncoder, Experiment


torch = pytest.importorskip("torch")
if not hasattr(torch, "Tensor"):
    sys.modules.pop("torch", None)
    pytest.skip("PyTorch is not installed.", allow_module_level=True)


def test_torch_basic_training_updates_experiment_state():
    from dryml.models.torch import Model, Optimizer, Training

    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32)
    ds = ArrayDataset((x, y))

    model = Model(torch.nn.Linear, 1, 1)
    optimizer = Optimizer(torch.optim.SGD, target=model, lr=0.01)
    train_fn = Training(
        optimizer=optimizer,
        loss_cls=torch.nn.MSELoss,
        epochs=2,
        batch_size=2,
        verbose=0,
    )
    exp = Experiment(model, train_fn, train_data=ds)

    losses = exp.train()

    assert len(losses) == 4
    assert exp.state.epoch == 2
    assert exp.state.step == 4
    assert exp.state.phase == "trained"
    assert optimizer.obj is not None
    assert optimizer.obj.param_groups[0]["lr"] == 0.01


def test_torch_sequential_accepts_constructor_tuple_shorthand():
    from dryml.models.torch import Sequential

    x = np.zeros((4, 3), dtype=np.float32)
    ds = ArrayDataset(x)
    model = Sequential(layer_defs=[
        ("Linear", 3, 8),
        ("ReLU",),
        ("Linear", 8, 2),
    ])

    assert Map(ds, model).spec.backend.value == "torch"
    assert Map(ds, model).spec.shape == (2,)


def test_torch_model_map_unbatched_tensor_uses_backend_batch_axis():
    from dryml.models.torch import Sequential

    x = np.zeros((2, 3, 4), dtype=np.float32)
    ds = ArrayDataset(x)
    model = Sequential(layer_defs=[
        ("Flatten",),
        ("Linear", 12, 2),
    ])
    mapped = Map(ds, model)

    out = list(mapped)

    assert mapped.spec.backend.value == "torch"
    assert mapped.spec.shape == (2,)
    assert [tuple(item.shape) for item in out] == [(2,), (2,)]


def test_torch_model_project_pipe_maps_unbatched_tensors_and_preserves_labels():
    from dryml.models.torch import Sequential

    x = np.zeros((2, 3, 4), dtype=np.float32)
    y = np.array([1, 0], dtype=np.int64)
    ds = ArrayDataset((x, y))
    model = Sequential(layer_defs=[
        ("Flatten",),
        ("Linear", 12, 2),
    ])
    mapped = Map(ds, Project(Pipe(Select(0), model), Select(1)))

    out = list(mapped)

    assert mapped.spec[0].backend.value == "torch"
    assert mapped.spec[0].shape == (2,)
    assert [tuple(latent.shape) for latent, _ in out] == [(2,), (2,)]
    assert [int(label) for _, label in out] == [1, 0]


def test_torch_argmax_pipeline_after_model_output():
    from dryml.models.torch import Sequential

    x = np.zeros((2, 3, 4), dtype=np.float32)
    y = np.array([1, 0], dtype=np.int64)
    ds = ArrayDataset((x, y))
    encoder = Sequential(layer_defs=[
        ("Flatten",),
        ("Linear", 12, 2),
    ])
    classifier = Sequential(layer_defs=[
        ("Linear", 2, 3),
    ])
    mapped = Map(ds, Project(Pipe(Select(0), encoder, classifier, ArgMax()), Select(1)))

    out = list(mapped)

    assert mapped.spec[0] == TensorSpec("int64", shape=(), backend="torch")
    assert [tuple(pred.shape) for pred, _ in out] == [(), ()]
    assert [int(label) for _, label in out] == [1, 0]


def test_torch_autoencoder_optimizer_targets_composite_model():
    from dryml.models.torch import Optimizer, Sequential, Training

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
    train_fn = Training(
        optimizer=optimizer,
        loss_cls=torch.nn.MSELoss,
        epochs=2,
        batch_size=2,
        verbose=0,
    )
    exp = Experiment(model, train_fn, train_data=ds)

    expected_params = sum(1 for _ in encoder.trainable_parameters("torch")) + sum(
        1 for _ in decoder.trainable_parameters("torch")
    )
    actual_params = sum(len(group["params"]) for group in optimizer.obj.param_groups)
    losses = exp.train()

    assert not hasattr(model, "trainable_parameters")
    assert actual_params == expected_params
    assert len(losses) == 4
    assert exp.state.phase == "trained"


def test_torch_optimizer_targets_pipe_graph_without_pipe_trainable_parameters():
    from dryml.models.torch import Optimizer, Sequential

    repo = Repo()
    model = Sequential(layer_defs=(("Linear", (3, 2), {}),), repo=repo)
    pipe = Pipe(model, repo=repo)
    optimizer = Optimizer(torch.optim.SGD, target=pipe, lr=0.05, repo=repo)

    expected_params = sum(1 for _ in model.trainable_parameters("torch"))
    actual_params = sum(len(group["params"]) for group in optimizer.obj.param_groups)

    assert not hasattr(pipe, "trainable_parameters")
    assert actual_params == expected_params
