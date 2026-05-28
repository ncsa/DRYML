from __future__ import annotations

import numpy as np

from dryml.core2.tensor_spec import iter_specs
from dryml.data import Batch, Map, Pack, Select


def _is_batched(dataset) -> bool:
    try:
        return any(spec.batched for spec in iter_specs(dataset.spec))
    except ValueError:
        return False


def _prediction_pairs(model, data, *, x_transform=None, y_transform=None, batch_size: int | None = None):
    x_transform = Select(0) if x_transform is None else x_transform
    y_transform = Select(1) if y_transform is None else y_transform

    x_data = Map(data, x_transform)
    y_data = Map(data, y_transform)

    if batch_size is not None:
        x_data = Batch(x_data, batch_size)
        y_data = Batch(y_data, batch_size)

    return Pack(Map(x_data, model), y_data)


def _example_count(value, *, batched: bool) -> int:
    arr = np.asarray(value)
    if batched and arr.ndim > 0:
        return int(arr.shape[0])
    return 1


def mean_squared_error(model, test_data, *, x_transform=None, y_transform=None, batch_size: int | None = None):
    pairs = _prediction_pairs(
        model,
        test_data,
        x_transform=x_transform,
        y_transform=y_transform,
        batch_size=batch_size,
    )
    batched = _is_batched(pairs)

    total_loss = 0.0
    num_examples = 0
    for y_pred, y_true in pairs:
        diff = np.asarray(y_pred) - np.asarray(y_true)
        total_loss += float(np.sum(diff * diff))
        num_examples += _example_count(y_true, batched=batched)

    if num_examples == 0:
        raise ValueError("Cannot compute mean_squared_error on an empty dataset.")
    return total_loss / num_examples


def _as_labels(value, *, batched: bool):
    arr = np.asarray(value)

    if arr.ndim == 0:
        return arr

    if batched and arr.ndim == 1:
        return arr

    if arr.shape[-1] > 1:
        return np.argmax(arr, axis=-1)
    return arr


def categorical_accuracy(model, test_data, *, x_transform=None, y_transform=None, batch_size: int | None = None):
    pairs = _prediction_pairs(
        model,
        test_data,
        x_transform=x_transform,
        y_transform=y_transform,
        batch_size=batch_size,
    )
    batched = _is_batched(pairs)

    num_correct = 0
    num_total = 0
    for y_pred, y_true in pairs:
        pred_labels = _as_labels(y_pred, batched=batched)
        true_labels = _as_labels(y_true, batched=batched)
        matches = np.asarray(pred_labels == true_labels)
        num_correct += int(np.sum(matches))
        num_total += int(matches.size)

    if num_total == 0:
        raise ValueError("Cannot compute categorical_accuracy on an empty dataset.")
    return num_correct / num_total


__all__ = ["categorical_accuracy", "mean_squared_error"]
