from __future__ import annotations

import numpy as np
import pytest

from dryml.artifacts import CachedDataset
from dryml.core2 import Repo
from dryml.core2.store.dir import DirStore
from dryml.data import ArrayDataset
from dryml.managed import (
    ControlRequest,
    ManagedCallback,
    ManagedCleanupRefusedError,
    ManagedInterruptedError,
    MissingManagedOutputError,
    export_recipe,
    plan_cleanup,
    transfer_realizations,
)
from dryml.models import Experiment

from managed_workflow_fixtures import (
    ResumableSyntheticTraining,
    build_managed_workflow,
    completed_cache,
)


def test_full_managed_analyst_and_collaborator_workflow(tmp_path):
    pytest.importorskip("pyarrow")
    source = DirStore(tmp_path / "source")
    model, train, test, experiment, accuracy, confusion = build_managed_workflow(source)
    interrupted_once = {"value": False}

    def interrupt_first_progress(event):
        if event.kind == "progress" and not interrupted_once["value"]:
            interrupted_once["value"] = True
            return ControlRequest.INTERRUPT
        return None

    callback = ManagedCallback(
        interrupt_first_progress,
        controls={ControlRequest.INTERRUPT},
    )
    with pytest.raises(ManagedInterruptedError):
        experiment.train(store=source, callbacks=(callback,))

    interrupted = experiment.train.status(store=source)
    assert interrupted.status == "interrupted"
    assert interrupted.checkpoint_head is not None
    trained = experiment.train(store=source)
    assert trained.action == "resume"
    assert experiment.trained_model(store=source).trained_steps == 3
    assert tuple(item.record_id for item in trained.consumed_records) == (
        train.compute.results(store=source)["data"].record_id,
        test.compute.results(store=source)["data"].record_id,
    )
    assert len(experiment.train.history(store=source)[0].attempt_ids) == 2

    accuracy_result = accuracy.compute(store=source)
    confusion_result = confusion.compute(store=source)
    assert accuracy.value(store=source) == pytest.approx(0.75)
    np.testing.assert_array_equal(
        confusion.matrix(store=source),
        ((1, 0, 0), (0, 1, 1), (0, 0, 1)),
    )
    assert accuracy.exists(store=source)
    assert confusion.compute(store=source).action == "reuse"

    adapter_cache = CachedDataset(ArrayDataset(np.arange(12).reshape(6, 2)))
    adapter_cache.compute(
        store=source,
        representation="numpy-sequence",
        shard_rows=2,
    )
    converted = adapter_cache.request_representation("parquet", store=source)
    assert converted.status == "ok"
    assert test.compute.results(store=source)["data"].record_id == (
        trained.consumed_records[1].record_id
    )

    fine_data = completed_cache(source, (0, 1), (0, 1))
    fine_tune = Experiment(
        model,
        ResumableSyntheticTraining(),
        train_data=fine_data,
        model_state=experiment.train.result,
    )
    fine_result = fine_tune.train(store=source)
    assert fine_tune.trained_model(store=source).trained_steps == 5
    assert tuple(item.record_id for item in fine_result.consumed_records) == (
        fine_data.compute.results(store=source)["data"].record_id,
        trained.outputs["model"].record_id,
    )

    recipe_store = DirStore(tmp_path / "recipe")
    export_recipe(confusion.definition, recipe_store, main=True)
    recipe_metric = Repo(recipe_store).load(confusion.definition, restore_state=False)
    with pytest.raises(MissingManagedOutputError, match="active"):
        recipe_metric.compute(store=recipe_store)

    exact_store = DirStore(tmp_path / "exact")
    metric_transfer = transfer_realizations(
        source, exact_store, confusion.compute.result
    )
    fine_transfer = transfer_realizations(
        source, exact_store, fine_tune.train.result
    )
    reopened = DirStore(exact_store.base_dir)
    loaded_metric = Repo(reopened).load(confusion.definition, restore_state=False)
    loaded_fine_tune = Repo(reopened).load(fine_tune.definition, restore_state=False)
    np.testing.assert_array_equal(
        loaded_metric.matrix(store=reopened), confusion.matrix(store=source)
    )
    assert loaded_metric.compute(store=reopened).action == "reuse"
    assert loaded_fine_tune.train(store=reopened).action == "reuse"
    assert loaded_fine_tune.trained_model(store=reopened).trained_steps == 5
    assert trained.outputs["model"].record_id in metric_transfer.records
    assert trained.outputs["model"].record_id in fine_transfer.records

    second = confusion.compute.rerun(store=source)
    confusion.compute.activate(confusion_result.realization_id, store=source)
    assert confusion.compute.status(store=source).active_realization_id == (
        confusion_result.realization_id
    )
    assert {item.realization_id for item in confusion.compute.history(store=source)} == {
        confusion_result.realization_id,
        second.realization_id,
    }
    with pytest.raises(ManagedCleanupRefusedError, match="active"):
        plan_cleanup(
            source,
            confusion.compute.result,
            realization_ids=(confusion_result.realization_id,),
        )

    assert accuracy_result.outputs["metric"].record_id != (
        confusion_result.outputs["metric"].record_id
    )


def test_compute_and_train_expose_the_same_managed_lifecycle_surface(tmp_path):
    store = DirStore(tmp_path / "store")
    _model, _train, _test, experiment, _accuracy, confusion = build_managed_workflow(store)
    experiment.train(store=store)
    confusion.compute(store=store)

    for operation in (experiment.train, confusion.compute):
        assert operation.status(store=store).status == "completed"
        assert operation.result is not None
        assert operation.results(store=store)
        assert operation.history(store=store)
        assert callable(operation.resume)
        assert callable(operation.rerun)
        assert callable(operation.activate)
