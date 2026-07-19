from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dryml.artifacts import CachedDataset
from dryml.core2 import Repo
from dryml.core2.store.dir import DirStore
from dryml.data import ArrayDataset
from dryml.managed import (
    ControlRequest,
    ManagedCallback,
    ManagedCapabilityError,
    ManagedInterruptedError,
    ManagedRerunRequiredError,
)
from dryml.models import Experiment, TrainResumeMode


torch = pytest.importorskip("torch")


class TrackingRegressor(torch.nn.Module):
    """Tiny stochastic module whose buffers expose exactly consumed row IDs."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((1, 1)))
        self.register_buffer("seen", torch.zeros(4, dtype=torch.int64))

    def forward(self, x):
        ids = x[:, 0].to(dtype=torch.int64)
        self.seen.index_add_(0, ids, torch.ones_like(ids))
        features = x[:, 1:2]
        keep = (torch.rand_like(features) >= 0.25).to(features.dtype)
        return features @ self.weight * keep


def _cache(store):
    ids = np.arange(4, dtype=np.float32)
    features = ids + 1.0
    x = np.stack((ids, features), axis=1)
    y = (2.0 * features)[:, None]
    cached = CachedDataset(ArrayDataset((x, y)))
    cached.compute(store=store, representation="numpy-sequence", shard_rows=2)
    return cached


def _experiment(store, *, epochs=3, **training_kwargs):
    from dryml.models.torch import Model, Training

    cached = _cache(store)
    model = Model(TrackingRegressor)
    trainer = Training(
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.05, "momentum": 0.8},
        loss_cls=torch.nn.MSELoss,
        epochs=epochs,
        batch_size=2,
        verbose=0,
        **training_kwargs,
    )
    return Experiment(model, trainer, train_data=cached), cached


def _interrupt_once():
    requested = False

    def callback(event):
        nonlocal requested
        if event.kind == "safe_point" and not requested:
            requested = True
            return ControlRequest.INTERRUPT

    return ManagedCallback(
        callback,
        controls={ControlRequest.INTERRUPT},
        fail_soft=True,
    )


def _checkpoint_root(store):
    roots = tuple(
        Path(store.managed_control_root()).glob(
            "operations/**/attempts/*/checkpoints/checkpoint-v1-*"
        )
    )
    torch_roots = tuple(root for root in roots if (root / "torch-train.json").exists())
    assert len(torch_roots) == 1
    return torch_roots[0]


def _control_snapshot(store):
    root = Path(store.managed_control_root())
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _trained_state(exp, store):
    return exp.trained_model(store=store).obj.state_dict()


def test_torch_training_advertises_only_supported_exact_pipeline():
    from dryml.models.torch import Training

    class CustomTraining(Training):
        pass

    exact = Training.resume_capability(
        Training(
            optimizer_cls=torch.optim.SGD,
            loss_cls=torch.nn.MSELoss,
            epochs=2,
        ).definition
    )
    shuffled = Training.resume_capability(
        Training(shuffle=True, shuffle_seed=7).definition
    )
    custom = CustomTraining.resume_capability(Training().definition)

    assert exact.mode is TrainResumeMode.EXACT
    assert exact.early_completion
    assert "RNG" in exact.diagnostic
    assert shuffled.mode is TrainResumeMode.NONE
    assert shuffled.checkpoint_schema is None
    assert "shuffle" in shuffled.diagnostic
    assert custom.mode is TrainResumeMode.NONE
    assert "custom" in custom.diagnostic


def test_torch_managed_training_completes_reuses_and_reloads_fresh_model(tmp_path):
    store = DirStore(tmp_path / "store")
    exp, _cached = _experiment(store, epochs=2)
    torch.manual_seed(41)

    completed = exp.train(store=store)
    reused = exp.train(store=store)
    reopened = Repo(store).load(exp.definition, restore_state=False)

    assert completed.action == "start"
    assert reused.action == "reuse"
    assert reused.realization_id == completed.realization_id
    assert reopened.train(store=store).action == "reuse"
    assert exp.train.status(store=store).progress.current == 2
    assert _trained_state(exp, store)["seen"].tolist() == [2, 2, 2, 2]
    assert exp.model.obj.seen.tolist() == [0, 0, 0, 0]


def test_torch_model_state_is_experiment_scoped_and_supports_fine_tuning(
    tmp_path,
):
    from dryml.models.torch import Training

    store = DirStore(tmp_path / "store")
    first, cached = _experiment(store, epochs=1)
    torch.manual_seed(19)
    first_result = first.train(store=store)
    fine_tune = Experiment(
        first.model,
        Training(
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.05, "momentum": 0.8},
            loss_cls=torch.nn.MSELoss,
            epochs=1,
            batch_size=2,
            verbose=0,
        ),
        train_data=cached,
        model_state=first.train.result,
    )

    fine_result = fine_tune.train(store=store)

    assert first_result.outputs["model"].record_id != fine_result.outputs["model"].record_id
    assert _trained_state(first, store)["seen"].tolist() == [1, 1, 1, 1]
    assert _trained_state(fine_tune, store)["seen"].tolist() == [2, 2, 2, 2]
    assert tuple(item.record_id for item in fine_result.consumed_records) == (
        cached.compute.results(store=store)["data"].record_id,
        first_result.outputs["model"].record_id,
    )


def test_torch_interruption_restores_model_optimizer_rng_progress_and_pinned_cache(
    tmp_path,
):
    resumed_store = DirStore(tmp_path / "resumed")
    baseline_store = DirStore(tmp_path / "baseline")
    resumed_exp, cached = _experiment(resumed_store)
    baseline_exp, _baseline_cache = _experiment(baseline_store)
    original_cache_id = cached.compute.results(store=resumed_store)["data"].record_id

    torch.manual_seed(73)
    with pytest.raises(ManagedInterruptedError):
        resumed_exp.train(store=resumed_store, callbacks=(_interrupt_once(),))

    interrupted = resumed_exp.train.status(store=resumed_store)
    checkpoint = _checkpoint_root(resumed_store)
    descriptor = json.loads((checkpoint / "torch-train.json").read_text())
    optimizer_state = torch.load(
        checkpoint / "optimizer.pth",
        map_location="cpu",
        weights_only=False,
    )
    replacement = cached.compute.rerun(
        store=resumed_store,
        representation="numpy-sequence",
        shard_rows=1,
    )

    torch.manual_seed(999)
    resumed = resumed_exp.train(store=resumed_store)
    torch.manual_seed(73)
    baseline_exp.train(store=baseline_store)

    assert interrupted.status == "interrupted"
    assert interrupted.progress.current == 1
    assert descriptor["completed_epoch"] == 1
    assert descriptor["completed_step"] == 2
    assert descriptor["target_epoch"] == 3
    assert optimizer_state["state"]
    assert (checkpoint / "rng-state.pkl").is_file()
    assert replacement.outputs["data"].record_id != original_cache_id
    assert resumed.action == "resume"
    assert tuple(item.record_id for item in resumed.consumed_records) == (
        original_cache_id,
    )
    assert len(resumed_exp.train.history(store=resumed_store)[0].attempt_ids) == 2
    resumed_state = _trained_state(resumed_exp, resumed_store)
    baseline_state = _trained_state(baseline_exp, baseline_store)
    assert resumed_state["seen"].tolist() == [3, 3, 3, 3]
    assert baseline_state["seen"].tolist() == [3, 3, 3, 3]
    torch.testing.assert_close(resumed_state["weight"], baseline_state["weight"])


def test_torch_unsafe_pipeline_rejects_strict_resume_before_train_mutation(tmp_path):
    store = DirStore(tmp_path / "store")
    exp, _cached = _experiment(store, shuffle=True, shuffle_seed=7)
    before = _control_snapshot(store)

    with pytest.raises(ManagedCapabilityError, match="strict callback"):
        exp.train(store=store, callbacks=(lambda event: None,))

    assert _control_snapshot(store) == before
    assert exp.train.status(store=store).status == "not_started"


def test_torch_experiment_metric_state_downgrades_exact_capability(tmp_path):
    store = DirStore(tmp_path / "store")
    base, cached = _experiment(store, epochs=1)
    exp = Experiment(
        base.model,
        base.train_fn,
        train_data=cached,
        metrics={"opaque": 1},
    )
    before = _control_snapshot(store)

    with pytest.raises(ManagedCapabilityError, match="strict callback"):
        exp.train(store=store, callbacks=(lambda event: None,))

    assert _control_snapshot(store) == before
    assert exp.train.status(store=store).status == "not_started"


def test_torch_unsafe_pipeline_interrupt_requires_explicit_rerun(tmp_path):
    store = DirStore(tmp_path / "store")
    exp, _cached = _experiment(store, epochs=2, shuffle=True, shuffle_seed=7)

    with pytest.raises(ManagedInterruptedError):
        exp.train(store=store, callbacks=(_interrupt_once(),))

    assert exp.train.status(store=store).checkpoint_head is None
    with pytest.raises(ManagedRerunRequiredError):
        exp.train(store=store)
    assert exp.train.rerun(store=store).action == "rerun"


def test_torch_strict_callback_failure_checkpoints_then_resumes(tmp_path):
    store = DirStore(tmp_path / "store")
    exp, _cached = _experiment(store, epochs=2)

    def fail_at_epoch(event):
        if event.kind == "safe_point":
            raise RuntimeError("strict Torch observer failed")

    with pytest.raises(Exception, match="strict Torch observer failed"):
        exp.train(store=store, callbacks=(ManagedCallback(fail_at_epoch),))

    failed = exp.train.status(store=store)
    resumed = exp.train(store=store)

    assert failed.status == "failed"
    assert failed.checkpoint_head is not None
    assert failed.progress.current == 1
    assert resumed.action == "resume"


def test_torch_graceful_stop_completes_declared_epoch_prefix(tmp_path):
    store = DirStore(tmp_path / "store")
    exp, _cached = _experiment(store, epochs=4)
    callback = ManagedCallback(
        lambda event: (
            ControlRequest.GRACEFUL_STOP if event.kind == "safe_point" else None
        ),
        controls={ControlRequest.GRACEFUL_STOP},
        fail_soft=True,
    )

    result = exp.train(store=store, callbacks=(callback,))

    assert result.early_completed
    assert exp.train.status(store=store).status == "completed"
    assert exp.train.status(store=store).progress.current == 1
    assert _trained_state(exp, store)["seen"].tolist() == [1, 1, 1, 1]


def test_torch_corrupt_checkpoint_fails_closed_and_allows_rerun(tmp_path):
    store = DirStore(tmp_path / "store")
    exp, _cached = _experiment(store, epochs=2)

    with pytest.raises(ManagedInterruptedError):
        exp.train(store=store, callbacks=(_interrupt_once(),))
    (_checkpoint_root(store) / "optimizer.pth").write_bytes(b"corrupt")

    with pytest.raises(Exception, match="integrity|manifest|digest"):
        exp.train(store=store)
    assert exp.train.rerun(store=store).action == "rerun"


def test_torch_multi_rank_managed_publication_is_rejected_before_mutation(
    tmp_path,
    monkeypatch,
):
    store = DirStore(tmp_path / "store")
    exp, _cached = _experiment(store, epochs=1)
    before = _control_snapshot(store)
    monkeypatch.setenv("DRYML_WORLD_SIZE", "2")

    with pytest.raises(ManagedCapabilityError, match="multi-rank"):
        exp.train(store=store)

    assert _control_snapshot(store) == before
    assert exp.train.status(store=store).status == "not_started"


def test_importing_torch_model_declarations_does_not_import_torch():
    script = "import sys; import dryml.models.torch; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True)
