"""Lightweight deterministic fixtures for managed workflow and docs tests."""

from __future__ import annotations

import numpy as np

from dryml.artifacts import CachedDataset
from dryml.core import Repo
from dryml.core.object import Pickleable
from dryml.data import ArrayDataset
from dryml.managed import current_operation_context
from dryml.metrics import CategoricalAccuracy, ConfusionMatrix
from dryml.models import Experiment, Model, TrainCapability, TrainFunction
from dryml.models.utils import advance_train_state


class SyntheticClassifier(Model, Pickleable):
    """Table-based classifier with pickle-backed lightweight model state."""

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float64)
        self.trained_steps = 0

    def __call__(self, x):
        return self.scores[np.asarray(x, dtype=np.int64)]


class ResumableSyntheticTraining(TrainFunction):
    """Exact generic trainer that checkpoints after each deterministic row."""

    __dryml_train_capability__ = TrainCapability.exact(
        "synthetic model, cursor, and progress are fully checkpointed"
    )

    def __call__(self, exp):
        rows = tuple(exp.train_data)
        start = int(exp.resume_payload or 0)
        context = current_operation_context()
        for index in range(start, len(rows)):
            exp.model.trained_steps += 1
            advance_train_state(exp, steps=1, phase=exp.state.training)
            context.progress(index + 1, total=len(rows), message="synthetic training")
            next_index = index + 1
            context.safe_point(
                checkpoint=lambda next_index=next_index: self.checkpoint(
                    exp, payload=next_index
                )
            )
        advance_train_state(exp, epochs=1)


def completed_cache(store, x, y):
    """Create one completed exact NumPy cache for aligned ``(x, y)`` rows."""

    cache = CachedDataset(ArrayDataset((np.asarray(x), np.asarray(y))))
    cache.compute(
        store=store,
        representation="numpy-sequence",
        shard_rows=2,
    )
    return cache


def build_managed_workflow(store):
    """Return a trained synthetic Experiment, test cache, and both metrics."""

    scores = (
        (0.9, 0.05, 0.05),
        (0.1, 0.2, 0.7),
        (0.1, 0.1, 0.8),
        (0.1, 0.8, 0.1),
    )
    train = completed_cache(store, (0, 1, 2), (0, 1, 2))
    test = completed_cache(store, (0, 1, 2, 3), (0, 1, 2, 1))
    model = SyntheticClassifier(scores)
    Repo(store).save_object(model, record_policy="none")
    experiment = Experiment(
        model,
        ResumableSyntheticTraining(),
        train_data=train,
        test_data=test,
    )
    accuracy = CategoricalAccuracy(
        experiment.train.result,
        test.compute.result,
        labels=(0, 1, 2),
    )
    confusion = ConfusionMatrix(
        experiment.train.result,
        test.compute.result,
        labels=(0, 1, 2),
    )
    return model, train, test, experiment, accuracy, confusion


__all__ = [
    "ResumableSyntheticTraining",
    "SyntheticClassifier",
    "build_managed_workflow",
    "completed_cache",
]
