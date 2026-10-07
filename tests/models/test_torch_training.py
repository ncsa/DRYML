import numpy as np
import pytest
import sys

from dryml import F
from dryml.core.tensor_spec import TensorSpec
from dryml.data import ArgMax, ArrayDataset, Batch, Map, Pipe, Project, Select
from dryml.core import Repo
from dryml.managed import ManagedConfig
from dryml.models import AutoEncoder, Experiment


torch = pytest.importorskip("torch")
if not hasattr(torch, "Tensor"):
    sys.modules.pop("torch", None)
    pytest.skip("PyTorch is not installed.", allow_module_level=True)


class EffectfulModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return value


def _tiny_tensorflow_pairs():
    """Yield fixed TensorFlow examples for the real cross-backend trainer path."""
    import tensorflow as tf

    yield tf.constant([0.0], dtype=tf.float32), tf.constant([0.0], dtype=tf.float32)
    yield tf.constant([1.0], dtype=tf.float32), tf.constant([2.0], dtype=tf.float32)


def test_torch_basic_training_updates_experiment_state():
    from dryml.core.repo import get_default_repo
    from dryml.models.torch import Model, Optimizer, Training

    assert get_default_repo() is None
    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32)
    ds = Batch(ArrayDataset((x, y)), 2)

    model = Model(torch.nn.Linear, 1, 1)
    optimizer = Optimizer(torch.optim.SGD, target=model, lr=0.01)
    train_fn = Training(
        optimizer=optimizer,
        loss_cls=torch.nn.MSELoss,
        epochs=2,
        verbose=0,
    )
    exp = Experiment(model, train_fn, train_data=ds)

    losses = train_fn(exp)

    assert len(losses) == 4
    assert exp.state.epoch == 2
    assert exp.state.step == 4
    assert exp.state.examples_seen == 8
    assert exp.state.phase is None
    assert optimizer.obj is not None
    assert optimizer.obj.param_groups[0]["lr"] == 0.01
    assert get_default_repo() is None


def test_torch_model_and_optimizer_state_ref_round_trip(tmp_path):
    from dryml.models.torch import Model, Optimizer, Training

    repo = Repo(stores=tmp_path)
    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32)
    ds = Batch(ArrayDataset((x, y), repo=repo), 2, repo=repo)
    model = Model(torch.nn.Linear, 1, 1, repo=repo)
    optimizer = Optimizer(
        torch.optim.SGD, target=model, lr=0.01, momentum=0.9, repo=repo
    )
    train_fn = Training(
        optimizer=optimizer,
        loss_cls=torch.nn.MSELoss,
        epochs=2,
        verbose=0,
        repo=repo,
    )
    exp = Experiment(model, train_fn, train_data=ds, repo=repo)
    exp.train(managed=ManagedConfig(state_repo=repo))
    expected_model = {
        key: value.detach().clone() for key, value in model.obj.state_dict().items()
    }
    expected_optimizer = optimizer.obj.state_dict()

    state = repo.save_object(exp, deep_capture=True)
    repo.close(flush=True)
    loaded = Repo(stores=tmp_path).load_state_ref(state, reuse_live="never")

    assert loaded.state.examples_seen == 8
    assert loaded.state.next_batch == 0
    torch.testing.assert_close(loaded.model.obj.state_dict(), expected_model)
    torch.testing.assert_close(
        loaded.train_fn.optimizer.obj.state_dict(), expected_optimizer
    )


def test_torch_sequential_accepts_explicit_factory_specs():
    from dryml.models.torch import Sequential

    x = np.zeros((4, 3), dtype=np.float32)
    ds = ArrayDataset(x)
    model = Sequential(layer_defs=[
        F("Linear", 3, 8),
        F("ReLU"),
        F("Linear", 8, 2),
    ])

    assert Map(ds, model).spec.backend.value == "torch"
    assert Map(ds, model).spec.shape == (2,)


def test_torch_sequential_rejects_shorthand_before_constructing_any_layer(monkeypatch):
    from dryml.models.torch import Sequential

    built = []

    def build(layer_def, **kwargs):
        built.append(layer_def)
        return torch.nn.ReLU()

    monkeypatch.setattr(F, "build", build)
    for layer_defs in (["ReLU"], [("Linear", 3, 2)], [["Linear", 3, 2]], [F("ReLU"), "ReLU"]):
        with pytest.raises(TypeError, match="explicit FactorySpec.*Use F"):
            Sequential(layer_defs=layer_defs)

    assert built == []


def test_torch_sequential_rejects_wrong_backend_layer_type():
    from dryml.models.torch import Sequential

    with pytest.raises(TypeError, match="FactorySpec built object"):
        Sequential(layer_defs=[F(object)])


def test_torch_sequential_factory_state_ref_round_trip(tmp_path):
    from dryml.models.torch import Sequential

    repo = Repo(stores=tmp_path)
    model = Sequential(layer_defs=(F("Linear", 1, 1),), repo=repo)
    value = torch.tensor([[2.0]], dtype=torch.float32)
    expected = model(value).detach().clone()

    state = repo.save_object(model, deep_capture=True)
    repo.close(flush=True)
    loaded = Repo(stores=tmp_path).load_state_ref(state, reuse_live="never")

    torch.testing.assert_close(loaded(value), expected)


def test_torch_inference_never_invokes_an_opaque_model_or_changes_mode():
    from dryml.models.torch import Model

    model = Model(EffectfulModule)
    model.obj.train(True)

    with pytest.raises(NotImplementedError, match="pass output_spec explicitly"):
        model.infer_output_spec(TensorSpec("float32", shape=(3,), backend="numpy"))

    assert model.obj.calls == 0
    assert model.obj.training is True


def test_torch_selected_element_and_batched_calls_preserve_batch_boundaries_and_cache():
    from dryml.models.torch import Sequential

    model = Sequential(layer_defs=(F("Linear", 3, 2),))
    element_spec = TensorSpec("float32", shape=(3,), backend="numpy")
    batch_spec = TensorSpec("float32", shape=(3,), batch=2, backend="numpy")
    element = model.find_implementation(input_spec=element_spec)
    batched = model.find_implementation(input_spec=batch_spec)

    assert tuple(element(np.zeros((3,), dtype=np.float32)).shape) == (2,)
    assert tuple(batched(np.zeros((2, 3), dtype=np.float32)).shape) == (2, 2)

    model.learn()
    direct = model(torch.zeros((1, 3), dtype=torch.float32))
    assert model.call_mode == "cached"
    assert tuple(model(torch.zeros((1, 3), dtype=torch.float32)).shape) == tuple(direct.shape)
    model.eager()
    assert model.call_mode == "eager"


def test_torch_model_map_unbatched_tensor_uses_backend_batch_axis():
    from dryml.models.torch import Sequential

    x = np.zeros((2, 3, 4), dtype=np.float32)
    ds = ArrayDataset(x)
    model = Sequential(layer_defs=[
        F("Flatten"),
        F("Linear", 12, 2),
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
        F("Flatten"),
        F("Linear", 12, 2),
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
        F("Flatten"),
        F("Linear", 12, 2),
    ])
    classifier = Sequential(layer_defs=[
        F("Linear", 2, 3),
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
    ds = Batch(ArrayDataset((x, x.copy())), 2)
    encoder = Sequential(
        layer_defs=(
            F("Linear", 3, 8),
            F("ReLU"),
            F("Linear", 8, 2),
        )
    )
    decoder = Sequential(
        layer_defs=(
            F("Linear", 2, 8),
            F("ReLU"),
            F("Linear", 8, 3),
        )
    )
    model = AutoEncoder(encoder=encoder, decoder=decoder)
    optimizer = Optimizer(torch.optim.SGD, target=model, lr=0.05)
    train_fn = Training(
        optimizer=optimizer,
        loss_cls=torch.nn.MSELoss,
        epochs=2,
        verbose=0,
    )
    exp = Experiment(model, train_fn, train_data=ds)

    expected_params = sum(1 for _ in encoder.trainable_parameters("torch")) + sum(
        1 for _ in decoder.trainable_parameters("torch")
    )
    actual_params = sum(len(group["params"]) for group in optimizer.obj.param_groups)
    losses = train_fn(exp)

    assert not hasattr(model, "trainable_parameters")
    assert actual_params == expected_params
    assert len(losses) == 4
    assert exp.state.phase is None


def test_torch_optimizer_targets_pipe_graph_without_pipe_trainable_parameters():
    from dryml.models.torch import Optimizer, Sequential

    repo = Repo()
    model = Sequential(layer_defs=(F("Linear", 3, 2),), repo=repo)
    pipe = Pipe(model, repo=repo)
    optimizer = Optimizer(torch.optim.SGD, target=pipe, lr=0.05, repo=repo)

    expected_params = sum(1 for _ in model.trainable_parameters("torch"))
    actual_params = sum(len(group["params"]) for group in optimizer.obj.param_groups)

    assert not hasattr(pipe, "trainable_parameters")
    assert actual_params == expected_params


def test_torch_training_restore_skips_completed_batch_and_keeps_exposure(tmp_path):
    from dryml.models.torch import Model, Optimizer, Training

    repo = Repo(stores=tmp_path)
    x = np.arange(4, dtype=np.float32).reshape(-1, 1)
    y = x * 2.0
    model = Model(torch.nn.Linear, 1, 1, repo=repo)
    exp = Experiment(
        model,
        Training(
            optimizer=Optimizer(torch.optim.SGD, target=model, lr=0.01, repo=repo),
            loss_cls=torch.nn.MSELoss,
            epochs=1,
            verbose=0,
            repo=repo,
        ),
        train_data=Batch(ArrayDataset((x, y), repo=repo), 2, repo=repo),
        repo=repo,
    )

    def pause():
        raise RuntimeError("pause")

    with pytest.raises(RuntimeError, match="pause"):
        exp.train_fn(exp, callbacks=(pause,))
    checkpoint = repo.save_object(exp, deep_capture=True)
    restored = Repo(stores=tmp_path).load_state_ref(checkpoint, reuse_live="never")

    restored.train_fn(restored)

    assert restored.state.step == 2
    assert restored.state.examples_seen == 4
    assert restored.state.epoch == 1
    assert restored.state.next_batch == 0


def _torch_accounting_experiment(*, repo=None, count=64, batch_size=1, lr=0.01):
    from dryml.models.torch import Model, Optimizer, Training

    torch.manual_seed(7104)
    x = np.arange(count, dtype=np.float32).reshape(-1, 1) / 10.0
    y = x * 2.0
    model = Model(torch.nn.Linear, 1, 1, repo=repo)
    return Experiment(
        model,
        Training(
            optimizer=Optimizer(torch.optim.SGD, target=model, lr=lr, repo=repo),
            loss_cls=torch.nn.MSELoss,
            epochs=1,
            verbose=0,
            repo=repo,
        ),
        train_data=Batch(ArrayDataset((x, y), repo=repo), batch_size, repo=repo),
        repo=repo,
    )


def test_torch_interrupt_at_step_32_resumes_exact_remaining_epoch(tmp_path):
    repo = Repo(stores=tmp_path)
    interrupted = _torch_accounting_experiment(repo=repo)
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

    baseline = _torch_accounting_experiment()
    baseline_losses = baseline.train_fn(baseline)

    assert seen == [(step, 0, step) for step in range(1, 33)]
    assert len(remaining_losses) == 32
    assert len(baseline_losses) == 64
    assert resumed.state.step == baseline.state.step == 64
    assert resumed.state.examples_seen == baseline.state.examples_seen == 64
    assert resumed.state.epoch == baseline.state.epoch == 1
    assert resumed.state.next_batch == baseline.state.next_batch == 0
    assert resumed.state.loss_denominator == baseline.state.loss_denominator == 64
    torch.testing.assert_close(resumed.model.obj.state_dict(), baseline.model.obj.state_dict())
    torch.testing.assert_close(
        resumed.train_fn.optimizer.obj.state_dict(), baseline.train_fn.optimizer.obj.state_dict()
    )


def test_torch_accepts_a_legacy_exhausted_saved_epoch_without_empty_data_error():
    exp = _torch_accounting_experiment(count=2, batch_size=1)
    exp.state.next_batch = 2
    exp.state.target_epoch = 1

    losses = exp.train_fn(exp)

    assert losses == []
    assert exp.state.epoch == 1
    assert exp.state.next_batch == 0
    assert exp.state.target_epoch is None


def test_torch_weighted_loss_uses_actual_short_final_batch_and_normalizes_before_callback():
    from dryml.models.torch import Model, Optimizer, Training

    x = np.zeros((81, 1), dtype=np.float32)
    y = np.concatenate((np.ones((64, 1), dtype=np.float32), np.full((17, 1), 3.0, dtype=np.float32)))
    model = Model(torch.nn.Linear, 1, 1)
    with torch.no_grad():
        model.obj.weight.zero_()
        model.obj.bias.zero_()
    exp = Experiment(
        model,
        Training(
            optimizer=Optimizer(torch.optim.SGD, target=model, lr=0.0),
            loss_cls=torch.nn.MSELoss,
            epochs=1,
            verbose=0,
        ),
        train_data=Batch(ArrayDataset((x, y)), 64),
    )
    observed = []

    exp.train_fn(exp, callbacks=(lambda: observed.append((exp.state.epoch, exp.state.next_batch)),))

    assert exp.state.examples_seen == 81
    assert exp.state.loss_denominator == 81
    assert exp.state.loss_numerator / exp.state.loss_denominator == pytest.approx((64 + 17 * 9) / 81)
    assert observed == [(0, 1), (1, 0)]


def test_torch_callback_and_loss_preflight_leave_training_objects_untouched():
    from dryml.models.torch import Model, Optimizer, Training

    model = Model(torch.nn.Linear, 1, 1)
    model.obj.eval()
    optimizer = Optimizer(torch.optim.SGD, target=model, lr=0.01)
    exp = Experiment(
        model,
        Training(optimizer=optimizer, loss_cls=torch.nn.MSELoss, epochs=1, verbose=0),
        train_data=Batch(ArrayDataset((np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32))), 1),
    )

    with pytest.raises(TypeError, match="callbacks"):
        exp.train_fn(exp, callbacks=(lambda: None, object()))
    assert model.obj.training is False
    assert exp.state.step == exp.state.examples_seen == 0
    assert not optimizer.obj.state

    rejecting = Training(
        optimizer=optimizer,
        loss_cls=torch.nn.MSELoss,
        loss_kwargs={"reduction": "sum"},
        epochs=1,
        verbose=0,
    )
    with pytest.raises(ValueError, match="reduction='mean'"):
        rejecting(exp)
    assert model.obj.training is False
    assert exp.state.step == exp.state.examples_seen == 0
    assert not optimizer.obj.state


def test_torch_training_prepares_cross_backend_data_before_model_invocation(monkeypatch):
    from dryml.models.torch import Model, Optimizer, Training
    from dryml.models.utils import TrainingPreparation

    events = []
    original_prepare = TrainingPreparation.prepare

    def prepared(self, *args, **kwargs):
        events.append("prepare")
        return original_prepare(self, *args, **kwargs)

    monkeypatch.setattr(TrainingPreparation, "prepare", prepared)
    model = Model(torch.nn.Linear, 1, 1)
    original_forward = model.obj.forward

    def record_forward(value):
        events.append("model")
        return original_forward(value)

    monkeypatch.setattr(model.obj, "forward", record_forward)
    exp = Experiment(
        model,
        Training(
            optimizer=Optimizer(torch.optim.SGD, target=model, lr=0.0),
            loss_cls=torch.nn.MSELoss,
            epochs=1,
            verbose=0,
        ),
        train_data=Batch(ArrayDataset((np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32))), 1),
    )

    exp.train_fn(exp)

    assert events[:2] == ["prepare", "model"]


def test_torch_training_prepares_tensorflow_data_through_its_retained_method_edge():
    """TensorFlow batches reach the real Torch loop through TrainingPreparation."""
    tf = pytest.importorskip("tensorflow")
    import dryml.tf
    from dryml.core import TensorSpec
    from dryml.core.cardinality import Cardinality
    from dryml.data import GeneratorDataset
    from dryml.models.torch import Model, Optimizer, Training

    model = Model(torch.nn.Linear, 1, 1)
    trainer = Training(
        optimizer=Optimizer(torch.optim.SGD, target=model, lr=0.0),
        loss_cls=torch.nn.MSELoss, epochs=1, verbose=0,
    )
    exp = Experiment(
        model, trainer,
        train_data=Batch(GeneratorDataset(
            _tiny_tensorflow_pairs, cardinality=Cardinality.finite(2),
            spec=(TensorSpec("float32", shape=(1,), backend="tf"), TensorSpec("float32", shape=(1,), backend="tf")),
        ), 1),
    )

    trainer(exp)

    assert [edge.adapter for edge in trainer.method_graph().conversion_edges] == ["tf_to_torch", "tf_to_torch"]
    assert all(parameter.grad is not None for parameter in model.obj.parameters())


def test_torch_final_callback_resume_runs_one_validation_postlude_without_an_update(monkeypatch):
    from dryml.models.torch import Model, Optimizer, Training
    import dryml.models.torch.base as torch_base

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

    monkeypatch.setattr(torch_base, "TrainingProgress", Progress)
    model = Model(torch.nn.Linear, 1, 1)
    trainer = Training(
        optimizer=Optimizer(torch.optim.SGD, target=model, lr=0.0),
        loss_cls=torch.nn.MSELoss, epochs=1, verbose=0,
    )
    data = Batch(ArrayDataset((np.zeros((2, 1), dtype=np.float32), np.zeros((2, 1), dtype=np.float32))), 1)
    exp = Experiment(model, trainer, train_data=data, val_data=data)
    evaluations = []
    monkeypatch.setattr(trainer, "_evaluate", lambda *args, **kwargs: evaluations.append("validation") or {})

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


def test_torch_training_plans_cross_backend_handoffs_once(monkeypatch):
    from dryml.models.torch import Model, Optimizer, Training
    import dryml.methods.conversion as conversion

    planned = []
    original_make_edge = conversion.make_edge

    def record_edge(*args, **kwargs):
        planned.append(args[0])
        return original_make_edge(*args, **kwargs)

    monkeypatch.setattr(conversion, "make_edge", record_edge)
    model = Model(torch.nn.Linear, 1, 1)
    trainer = Training(
        optimizer=Optimizer(torch.optim.SGD, target=model, lr=0.0),
        loss_cls=torch.nn.MSELoss, epochs=1, verbose=0,
    )
    exp = Experiment(
        model, trainer,
        train_data=Batch(ArrayDataset((np.zeros((3, 1), dtype=np.float32), np.zeros((3, 1), dtype=np.float32))), 1),
    )

    trainer(exp)

    assert len(planned) == 2
    assert len(trainer.training_preparation.conversion_edges) == 2
    assert all(edge is not None for edge in trainer.training_preparation.conversion_edges)
    assert trainer.training_preparation.consumer_specs[0].backend.value == "torch"
    assert trainer.method_graph().conversion_edges == trainer.training_preparation.conversion_edges


def test_torch_unknown_stream_retains_validation_metrics_across_progress_retry(monkeypatch):
    from dryml.models.torch import Model, Optimizer, Training
    import dryml.models.torch.base as torch_base

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

    monkeypatch.setattr(torch_base, "TrainingProgress", Progress)
    model = Model(torch.nn.Linear, 1, 1)
    trainer = Training(
        optimizer=Optimizer(torch.optim.SGD, target=model, lr=0.0),
        loss_cls=torch.nn.MSELoss, epochs=1, verbose=0,
    )
    data = Batch(UnknownArrayDataset((np.zeros((2, 1), dtype=np.float32), np.zeros((2, 1), dtype=np.float32))), 1)
    exp = Experiment(model, trainer, train_data=data, val_data=data)
    evaluations = []
    monkeypatch.setattr(trainer, "_evaluate", lambda *args, **kwargs: evaluations.append("validation") or {"loss": 3.0})

    with pytest.raises(RuntimeError, match="progress postlude"):
        trainer(exp)
    assert evaluations == ["validation"]
    retained_metrics = exp.state.pending_epoch_metrics
    assert retained_metrics is not None
    assert retained_metrics["val_loss"] == 3.0

    trainer(exp)
    assert evaluations == ["validation"]
    assert Progress.received == [retained_metrics, retained_metrics]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"monitor": ""}, ValueError),
        ({"monitor": "accuracy"}, ValueError),
        ({"patience": True}, TypeError),
        ({"patience": -1}, ValueError),
        ({"mode": "auto"}, ValueError),
        ({"min_delta": -1.0}, ValueError),
        ({"min_delta": float("nan")}, ValueError),
    ],
)
def test_torch_early_stopping_rejects_invalid_configuration(kwargs, error):
    from dryml.models.torch import EarlyStoppingTraining, Model, Optimizer

    model = Model(torch.nn.Linear, 1, 1)
    optimizer = Optimizer(torch.optim.SGD, target=model, lr=0.0)
    with pytest.raises(error):
        EarlyStoppingTraining(
            optimizer=optimizer,
            loss_cls=torch.nn.MSELoss,
            epochs=3,
            verbose=0,
            **kwargs,
        )


def test_torch_early_stopping_accepts_shortened_target_and_retains_decision():
    from dryml.models.torch import EarlyStoppingTraining, Model, Optimizer

    data = Batch(ArrayDataset((
        np.zeros((2, 1), dtype=np.float32),
        np.ones((2, 1), dtype=np.float32),
    )), 1)
    model = Model(torch.nn.Linear, 1, 1)
    trainer = EarlyStoppingTraining(
        optimizer=Optimizer(torch.optim.SGD, target=model, lr=0.0),
        loss_cls=torch.nn.MSELoss,
        epochs=5,
        monitor="loss",
        patience=0,
        mode="min",
        min_delta=0.0,
        restore_best_weights=False,
        verbose=0,
    )
    exp = Experiment(model, trainer, train_data=data)

    trainer(exp)

    assert exp.state.epoch == 2
    assert exp.state.target_epoch is None
    assert trainer.early_stopping.best_epoch == 0
    assert trainer.early_stopping.wait == 1
    assert trainer.early_stopping.accepted_target == 2


def test_torch_early_stopping_completes_full_target_before_patience():
    from dryml.models.torch import EarlyStoppingTraining, Model, Optimizer

    data = Batch(ArrayDataset((
        np.zeros((2, 1), dtype=np.float32),
        np.ones((2, 1), dtype=np.float32),
    )), 1)
    model = Model(torch.nn.Linear, 1, 1)
    trainer = EarlyStoppingTraining(
        optimizer=Optimizer(torch.optim.SGD, target=model, lr=0.0),
        loss_cls=torch.nn.MSELoss,
        epochs=2,
        monitor="loss",
        patience=2,
        restore_best_weights=False,
        verbose=0,
    )
    exp = Experiment(model, trainer, train_data=data)

    trainer(exp)

    assert exp.state.epoch == 2
    assert exp.state.target_epoch is None
    assert trainer.early_stopping.accepted_target is None


@pytest.mark.parametrize(
    ("validation_metrics", "match"),
    [({}, "missing"), ({"loss": float("nan")}, "finite")],
)
def test_torch_early_stopping_rejects_missing_or_nonfinite_monitor(
    monkeypatch, validation_metrics, match,
):
    from dryml.models.torch import EarlyStoppingTraining, Model, Optimizer

    data = Batch(ArrayDataset((
        np.zeros((1, 1), dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
    )), 1)
    model = Model(torch.nn.Linear, 1, 1)
    trainer = EarlyStoppingTraining(
        optimizer=Optimizer(torch.optim.SGD, target=model, lr=0.0),
        loss_cls=torch.nn.MSELoss,
        epochs=2,
        monitor="val_loss",
        patience=0,
        verbose=0,
    )
    exp = Experiment(model, trainer, train_data=data, val_data=data)
    monkeypatch.setattr(trainer, "_evaluate", lambda *args, **kwargs: validation_metrics)

    with pytest.raises(ValueError, match=match):
        trainer(exp)

    assert exp.state.step == 1
    assert exp.state.target_epoch == 2


def test_torch_early_stopping_restores_only_best_model_state(monkeypatch):
    from dryml.models.torch import EarlyStoppingTraining, Model, Optimizer

    data = Batch(ArrayDataset((
        np.ones((2, 1), dtype=np.float32),
        np.zeros((2, 1), dtype=np.float32),
    )), 2)
    model = Model(torch.nn.Linear, 1, 1)
    optimizer = Optimizer(torch.optim.Adam, target=model, lr=0.1)
    trainer = EarlyStoppingTraining(
        optimizer=optimizer,
        loss_cls=torch.nn.MSELoss,
        epochs=5,
        monitor="val_loss",
        patience=0,
        restore_best_weights=True,
        verbose=0,
    )
    exp = Experiment(model, trainer, train_data=data, val_data=data)
    validation = iter((1.0, 2.0))
    monkeypatch.setattr(
        trainer,
        "_evaluate",
        lambda *args, **kwargs: {"loss": next(validation)},
    )
    observed = {}
    original = trainer._finish_training_behavior_epoch

    def finish(exp, epoch, metrics):
        observed[epoch] = {
            name: value.detach().clone() for name, value in model.obj.state_dict().items()
        }
        return original(exp, epoch, metrics)

    monkeypatch.setattr(trainer, "_finish_training_behavior_epoch", finish)

    trainer(exp)

    torch.testing.assert_close(model.obj.state_dict(), observed[0])
    assert exp.state.epoch == 2
    assert next(iter(optimizer.obj.state.values()))["step"].item() == 2
    assert trainer.early_stopping.restored is True


def test_torch_early_stopping_postlude_resume_retains_one_decision(monkeypatch, tmp_path):
    from dryml.models.torch import EarlyStoppingTraining, Model, Optimizer
    import dryml.models.torch.base as torch_base

    class Progress:
        attempts = 0

        def __init__(self, **kwargs):
            del kwargs

        def update(self, *args, **kwargs):
            del args, kwargs

        def epoch_end(self, *args, **kwargs):
            del args, kwargs
            type(self).attempts += 1
            if type(self).attempts == 2:
                raise RuntimeError("progress postlude")

        def close(self):
            pass

    monkeypatch.setattr(torch_base, "TrainingProgress", Progress)
    data = Batch(ArrayDataset((
        np.zeros((2, 1), dtype=np.float32),
        np.ones((2, 1), dtype=np.float32),
    )), 1)
    model = Model(torch.nn.Linear, 1, 1)
    trainer = EarlyStoppingTraining(
        optimizer=Optimizer(torch.optim.SGD, target=model, lr=0.0),
        loss_cls=torch.nn.MSELoss,
        epochs=5,
        monitor="loss",
        patience=0,
        restore_best_weights=True,
        verbose=0,
    )
    exp = Experiment(model, trainer, train_data=data)

    with pytest.raises(RuntimeError, match="progress postlude"):
        trainer(exp)
    assert trainer.early_stopping.wait == 1
    assert exp.state.pending_epoch_postlude == 1

    repo = Repo(stores=tmp_path)
    checkpoint = repo.save_object(exp, deep_capture=True)
    resumed = repo.load_state_ref(checkpoint, reuse_live="never")
    resumed.train_fn(resumed)

    assert resumed.train_fn.early_stopping.wait == 1
    assert resumed.state.target_epoch is None


def test_torch_managed_early_stopping_returns_exact_terminal_state(tmp_path):
    """Managed completion associates the shortened mixed-point graph itself."""

    from dryml.models.torch import EarlyStoppingTraining, Model, Optimizer

    repo = Repo(stores=tmp_path)
    data = Batch(ArrayDataset((
        np.zeros((2, 1), dtype=np.float32),
        np.ones((2, 1), dtype=np.float32),
    )), 1)
    model = Model(torch.nn.Linear, 1, 1)
    trainer = EarlyStoppingTraining(
        optimizer=Optimizer(torch.optim.SGD, target=model, lr=0.0),
        loss_cls=torch.nn.MSELoss,
        epochs=5,
        monitor="loss",
        patience=0,
        restore_best_weights=True,
        verbose=0,
    )
    exp = Experiment(model, trainer, train_data=data)

    final = exp.train(managed=ManagedConfig(state_repo=repo))
    status = exp.train.status(state_repo=repo)
    restored = repo.load_state_ref(final, reuse_live="never")

    assert final == status.final_state_ref == exp.last_state_ref
    assert (restored.state.epoch, restored.state.step, restored.state.target_epoch) == (2, 4, None)
    assert restored.train_fn.early_stopping.accepted_target == 2
    assert restored.train_fn.early_stopping.restored is True
