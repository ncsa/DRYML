from __future__ import annotations

import numpy as np
import pytest

from dryml.artifacts import CachedDataset
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.data import ArrayDataset
from dryml.managed import (
    ControlRequest,
    ManagedCallback,
    ManagedInterruptedError,
    ManagedRerunRequiredError,
    current_operation_context,
)
from dryml.models import (
    Experiment,
    Model,
    TrainCapability,
    TrainFunction,
    TrainResumeMode,
)


class ResumeModel(Model, Pickleable):
    def __init__(self):
        self.values = []


class ExactTraining(TrainFunction):
    __dryml_train_capability__ = TrainCapability.exact("test exact trainer")

    calls = 0

    def __call__(self, exp):
        type(self).calls += 1
        payload = exp.resume_payload or {"next": 0}
        rows = [int(np.asarray(row)[0]) for row in exp.train_data]
        if payload["next"] == 0:
            exp.model.values.append(rows[0])
            payload = {"next": 1}
            current_operation_context().safe_point(
                checkpoint=lambda: self.checkpoint(exp, payload=payload)
            )
        exp.model.values.extend(rows[payload["next"]:])


class OpaqueTraining(TrainFunction):
    def __call__(self, exp):
        current_operation_context().safe_point()
        exp.model.values.extend(int(np.asarray(row)[0]) for row in exp.train_data)


def _experiment(store, trainer):
    cached = CachedDataset(ArrayDataset(np.array([[1], [2], [3]], dtype=np.int64)))
    cached.compute(store=store, representation="numpy-sequence", shard_rows=1)
    return Experiment(ResumeModel(), trainer, train_data=cached), cached


def _interrupt_once():
    interrupted = {"done": False}

    def callback(event):
        if event.kind == "safe_point" and not interrupted["done"]:
            interrupted["done"] = True
            return ControlRequest.INTERRUPT

    return ManagedCallback(
        callback,
        controls={ControlRequest.INTERRUPT},
        fail_soft=True,
    )


def test_train_capability_contract_is_explicit_and_definition_driven():
    exact = ExactTraining.resume_capability(ExactTraining().definition)
    opaque = OpaqueTraining.resume_capability(OpaqueTraining().definition)

    assert exact.mode is TrainResumeMode.EXACT
    assert exact.checkpoint_schema == "dryml.experiment-train.v1"
    assert opaque.mode is TrainResumeMode.NONE
    assert opaque.checkpoint_schema is None


def test_interrupted_exact_training_resumes_same_inputs_and_checkpoint(tmp_path):
    store = DirStore(tmp_path / "store")
    exp, cached = _experiment(store, ExactTraining())
    cache_record = cached.compute.results(store=store)["data"].record_id
    ExactTraining.calls = 0

    with pytest.raises(ManagedInterruptedError):
        exp.train(store=store, callbacks=(_interrupt_once(),))

    interrupted = exp.train.status(store=store)
    assert interrupted.status == "interrupted"
    assert interrupted.checkpoint_head is not None
    replacement = cached.compute.rerun(store=store, representation="numpy-sequence")
    assert replacement.outputs["data"].record_id != cache_record
    resumed = exp.train(store=store)

    assert resumed.action == "resume"
    assert ExactTraining.calls == 2
    assert tuple(item.record_id for item in resumed.consumed_records) == (cache_record,)
    assert exp.trained_model(store=store).values == [1, 2, 3]
    assert len(exp.train.history(store=store)[0].attempt_ids) == 2


def test_incapable_training_requires_explicit_rerun_after_interrupt(tmp_path):
    store = DirStore(tmp_path / "store")
    exp, _cached = _experiment(store, OpaqueTraining())

    with pytest.raises(ManagedInterruptedError):
        exp.train(store=store, callbacks=(_interrupt_once(),))

    assert exp.train.status(store=store).checkpoint_head is None
    with pytest.raises(ManagedRerunRequiredError):
        exp.train(store=store)

    rerun = exp.train.rerun(store=store)
    assert rerun.action == "rerun"
    assert exp.trained_model(store=store).values == [1, 2, 3]
