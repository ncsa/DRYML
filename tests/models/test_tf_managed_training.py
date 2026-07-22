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


tf = pytest.importorskip("tensorflow")


def test_tf_checkpoint_staging_copies_complete_payload(tmp_path):
    from dryml.models.tf.base import _stage_tf_checkpoint

    prefix = tmp_path / "source" / "ckpt"
    prefix.parent.mkdir()
    prefix.with_suffix(".index").write_bytes(b"index")
    prefix.with_suffix(".data-00000-of-00001").write_bytes(b"data")

    staged, directory = _stage_tf_checkpoint(prefix)
    try:
        assert staged.with_suffix(".index").read_bytes() == b"index"
        assert staged.with_suffix(".data-00000-of-00001").read_bytes() == b"data"
    finally:
        directory.cleanup()


def _cache(store, values=(1.0, 2.0, 3.0, 4.0)):
    x = np.asarray(values, dtype=np.float32)[:, None]
    y = 2.0 * x
    cached = CachedDataset(ArrayDataset((x, y)))
    cached.compute(store=store, representation="numpy-sequence", shard_rows=2)
    return cached


def _experiment(store, *, epochs=3, **training_kwargs):
    from dryml.models.tf import BasicTraining, Loss, Optimizer, Sequential

    model = Sequential(
        layer_defs=((
            "Dense",
            1,
            {
                "use_bias": False,
                "kernel_initializer": "zeros",
            },
        ),)
    )
    trainer = BasicTraining(
        optimizer=Optimizer(
            tf.keras.optimizers.SGD,
            learning_rate=0.05,
            momentum=0.8,
        ),
        loss=Loss(tf.keras.losses.MeanSquaredError),
        epochs=epochs,
        batch_size=2,
        verbose=0,
        **training_kwargs,
    )
    cached = _cache(store)
    return Experiment(model, trainer, train_data=cached), cached


def _weights(exp, store):
    model = exp.trained_model(store=store)
    model.obj(np.zeros((1, 1), dtype=np.float32), training=False)
    return tuple(np.array(value, copy=True) for value in model.obj.get_weights())


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
    train_roots = tuple(root for root in roots if (root / "tensorflow-train.json").exists())
    assert len(train_roots) == 1
    return train_roots[0]


def _control_snapshot(store):
    root = Path(store.managed_control_root())
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_tf_basic_training_advertises_only_epoch_boundary_exact_pipelines():
    from dryml.models.tf import BasicTraining, Wrapper

    exact = BasicTraining.resume_capability(BasicTraining(epochs=2).definition)
    shuffled = BasicTraining.resume_capability(
        BasicTraining(epochs=2, shuffle=True, shuffle_seed=7).definition
    )
    partial_epoch = BasicTraining.resume_capability(
        BasicTraining(epochs=2, fit_kwargs={"steps_per_epoch": 1}).definition
    )
    configured_callback = BasicTraining.resume_capability(
        BasicTraining(
            epochs=2,
            callbacks=(Wrapper(tf.keras.callbacks.TerminateOnNaN),),
        ).definition
    )

    assert exact.mode is TrainResumeMode.EXACT
    assert exact.early_completion
    assert "epoch" in exact.diagnostic
    for capability in (shuffled, partial_epoch, configured_callback):
        assert capability.mode is TrainResumeMode.NONE
        assert capability.checkpoint_schema is None


def test_tf_managed_training_completes_reuses_and_reloads_fresh_model(tmp_path):
    store = DirStore(tmp_path / "store")
    exp, _cached = _experiment(store, epochs=2)

    completed = exp.train(store=store)
    reused = exp.train(store=store)
    reopened = Repo(store).load(exp.definition, restore_state=False)

    assert completed.action == "start"
    assert reused.action == "reuse"
    assert reused.realization_id == completed.realization_id
    assert reopened.train(store=store).action == "reuse"
    assert _weights(reopened, store)[0][0, 0] != 0.0
    assert exp.train.status(store=store).progress.current == 2


def test_tf_interruption_restores_model_optimizer_progress_and_pinned_cache(tmp_path):
    resumed_store = DirStore(tmp_path / "resumed")
    baseline_store = DirStore(tmp_path / "baseline")
    resumed_exp, cached = _experiment(resumed_store)
    baseline_exp, _baseline_cache = _experiment(baseline_store)
    original_cache_id = cached.compute.results(store=resumed_store)["data"].record_id

    with pytest.raises(ManagedInterruptedError):
        resumed_exp.train(store=resumed_store, callbacks=(_interrupt_once(),))

    interrupted = resumed_exp.train.status(store=resumed_store)
    descriptor = json.loads(
        (_checkpoint_root(resumed_store) / "tensorflow-train.json").read_text(
            encoding="utf-8"
        )
    )
    replacement = cached.compute.rerun(
        store=resumed_store,
        representation="numpy-sequence",
        shard_rows=1,
    )
    resumed = resumed_exp.train(store=resumed_store)
    baseline_exp.train(store=baseline_store)

    assert interrupted.status == "interrupted"
    assert interrupted.progress.current == 1
    assert descriptor["completed_epoch"] == 1
    assert descriptor["completed_step"] == 2
    assert descriptor["target_epoch"] == 3
    assert replacement.outputs["data"].record_id != original_cache_id
    assert resumed.action == "resume"
    assert tuple(item.record_id for item in resumed.consumed_records) == (
        original_cache_id,
    )
    assert len(resumed_exp.train.history(store=resumed_store)[0].attempt_ids) == 2
    for actual, expected in zip(
        _weights(resumed_exp, resumed_store),
        _weights(baseline_exp, baseline_store),
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_tf_unsafe_pipeline_rejects_strict_resume_guarantee_before_train_mutation(
    tmp_path,
):
    store = DirStore(tmp_path / "store")
    exp, _cached = _experiment(store, shuffle=True, shuffle_seed=7)
    before = _control_snapshot(store)

    with pytest.raises(ManagedCapabilityError, match="strict callback"):
        exp.train(store=store, callbacks=(lambda event: None,))

    assert _control_snapshot(store) == before
    assert exp.train.status(store=store).status == "not_started"


def test_tf_pipeline_without_optimizer_downgrades_before_train_mutation(tmp_path):
    from dryml.models.tf import BasicTraining, Sequential

    store = DirStore(tmp_path / "store")
    cached = _cache(store)
    exp = Experiment(
        Sequential(layer_defs=(("Dense", 1),)),
        BasicTraining(epochs=1, batch_size=2, verbose=0),
        train_data=cached,
    )
    before = _control_snapshot(store)

    with pytest.raises(ManagedCapabilityError, match="strict callback"):
        exp.train(store=store, callbacks=(lambda event: None,))

    assert _control_snapshot(store) == before
    assert exp.train.status(store=store).status == "not_started"


def test_tf_unsafe_pipeline_interrupts_without_checkpoint_and_requires_rerun(
    tmp_path,
):
    store = DirStore(tmp_path / "store")
    exp, _cached = _experiment(store, epochs=2, shuffle=True, shuffle_seed=7)

    with pytest.raises(ManagedInterruptedError):
        exp.train(store=store, callbacks=(_interrupt_once(),))

    assert exp.train.status(store=store).checkpoint_head is None
    with pytest.raises(ManagedRerunRequiredError):
        exp.train(store=store)
    assert exp.train.rerun(store=store).action == "rerun"


def test_tf_strict_callback_failure_checkpoints_then_resumes(tmp_path):
    store = DirStore(tmp_path / "store")
    exp, _cached = _experiment(store, epochs=2)

    def fail_at_epoch(event):
        if event.kind == "safe_point":
            raise RuntimeError("strict TensorFlow observer failed")

    with pytest.raises(Exception, match="strict TensorFlow observer failed"):
        exp.train(store=store, callbacks=(ManagedCallback(fail_at_epoch),))

    failed = exp.train.status(store=store)
    resumed = exp.train(store=store)

    assert failed.status == "failed"
    assert failed.checkpoint_head is not None
    assert failed.progress.current == 1
    assert resumed.action == "resume"


def test_tf_strict_completion_failure_with_no_remaining_epochs_is_resumable(
    tmp_path,
):
    store = DirStore(tmp_path / "store")
    exp, _cached = _experiment(store, epochs=0)

    def fail_at_completion(event):
        if event.kind == "completed":
            raise RuntimeError("strict completion observer failed")

    with pytest.raises(Exception, match="strict completion observer failed"):
        exp.train(store=store, callbacks=(ManagedCallback(fail_at_completion),))

    failed = exp.train.status(store=store)
    assert failed.status == "failed"
    assert failed.checkpoint_head is not None
    assert exp.train(store=store).action == "resume"


def test_tf_fail_soft_observer_continues_and_reports_diagnostic(tmp_path):
    store = DirStore(tmp_path / "store")
    exp, _cached = _experiment(store, epochs=1)
    callback = ManagedCallback(
        lambda event: (_ for _ in ()).throw(RuntimeError("optional observer")),
        fail_soft=True,
    )

    result = exp.train(store=store, callbacks=(callback,))

    assert result.action == "start"
    assert result.diagnostics
    assert "optional observer" in result.diagnostics[0]
    assert exp.train.status(store=store).status == "completed"


def test_tf_graceful_early_stop_completes_declared_epoch_prefix(tmp_path):
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
    assert _weights(exp, store)[0][0, 0] != 0.0


def test_tf_missing_checkpoint_payload_fails_closed_and_allows_rerun(tmp_path):
    store = DirStore(tmp_path / "store")
    exp, _cached = _experiment(store, epochs=2)

    with pytest.raises(ManagedInterruptedError):
        exp.train(store=store, callbacks=(_interrupt_once(),))
    (_checkpoint_root(store) / "tensorflow-train.json").unlink()

    with pytest.raises(Exception, match="integrity|manifest|missing"):
        exp.train(store=store)

    rerun = exp.train.rerun(store=store)
    assert rerun.action == "rerun"


def test_tf_checkpoint_rejects_backend_version_mismatch(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    exp, _cached = _experiment(store, epochs=2)

    with pytest.raises(ManagedInterruptedError):
        exp.train(store=store, callbacks=(_interrupt_once(),))

    monkeypatch.setattr(tf, "__version__", "0.0-incompatible")
    with pytest.raises(ManagedCapabilityError, match="TensorFlow.*version"):
        exp.train(store=store)


def test_importing_tf_model_declarations_does_not_import_tensorflow():
    script = "import sys; import dryml.models.tf; assert 'tensorflow' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True)
