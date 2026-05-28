from __future__ import annotations

import numpy as np

from dryml.core2.tensor_spec import iter_specs
from dryml.data import Batch, Collect, iter_xy


def _is_batched(dataset) -> bool:
    try:
        return any(spec.batched for spec in iter_specs(dataset.spec))
    except ValueError:
        return False


def _prediction_pairs(model, data, *, x_path=0, y_path=1, batch_size: int | None = None):
    if batch_size is not None:
        data = Batch(data, batch_size)

    for x, y in iter_xy(data, x_path=x_path, y_path=y_path):
        yield model(x), y


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)



def _example_count(value, *, batched: bool) -> int:
    arr = _to_numpy(value)
    if batched and arr.ndim > 0:
        return int(arr.shape[0])
    return 1


def mean_squared_error(model, test_data, *, x_path=0, y_path=1, batch_size: int | None = None):
    pairs = _prediction_pairs(
        model,
        test_data,
        x_path=x_path,
        y_path=y_path,
        batch_size=batch_size,
    )
    batched = batch_size is not None or _is_batched(test_data)

    def step(acc, pair):
        total_loss, num_examples = acc
        y_pred, y_true = pair
        diff = _to_numpy(y_pred) - _to_numpy(y_true)
        return (
            total_loss + float(np.sum(diff * diff)),
            num_examples + _example_count(y_true, batched=batched),
        )

    total_loss, num_examples = Collect(step, initial=(0.0, 0))(pairs)
    if num_examples == 0:
        raise ValueError("Cannot compute mean_squared_error on an empty dataset.")
    return total_loss / num_examples


def _as_labels(value, *, batched: bool):
    arr = _to_numpy(value)

    if arr.ndim == 0:
        return arr

    if batched and arr.ndim == 1:
        return arr

    if arr.shape[-1] > 1:
        return np.argmax(arr, axis=-1)
    return arr


def categorical_accuracy(model, test_data, *, x_path=0, y_path=1, batch_size: int | None = None):
    pairs = _prediction_pairs(
        model,
        test_data,
        x_path=x_path,
        y_path=y_path,
        batch_size=batch_size,
    )
    batched = batch_size is not None or _is_batched(test_data)

    def step(acc, pair):
        num_correct, num_total = acc
        y_pred, y_true = pair
        pred_labels = _as_labels(y_pred, batched=batched)
        true_labels = _as_labels(y_true, batched=batched)
        matches = _to_numpy(pred_labels == true_labels)
        return num_correct + int(np.sum(matches)), num_total + int(matches.size)

    num_correct, num_total = Collect(step, initial=(0, 0))(pairs)
    if num_total == 0:
        raise ValueError("Cannot compute categorical_accuracy on an empty dataset.")
    return num_correct / num_total


__all__ = ["categorical_accuracy", "mean_squared_error"]
