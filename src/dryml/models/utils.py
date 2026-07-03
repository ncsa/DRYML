from __future__ import annotations

from typing import Any

from dryml.core2.tensor_spec import iter_specs
from dryml.data import Shuffle, Take, Unbatch
from dryml.models.train_spec import TrainState

def validate_num_examples(num_examples: int | None) -> None:
    if num_examples is not None and num_examples < 0:
        raise ValueError("num_examples must be non-negative or None.")


def dataset_is_batched(dataset) -> bool:
    try:
        return any(spec.batched for spec in iter_specs(dataset.spec))
    except ValueError:
        return False


def finite_dataset_len(dataset) -> int | None:
    try:
        cardinality = dataset.__len__()
    except Exception:
        return None

    if hasattr(cardinality, "is_finite"):
        if cardinality.is_finite:
            return cardinality.require_finite()
        return None
    return int(cardinality)


def prepare_training_data(
    train_data,
    *,
    num_examples: int | None = None,
    shuffle: bool = False,
    shuffle_seed=None,
    shuffle_buffer_size: int | None = None,
):
    if train_data is None:
        raise ValueError("Experiment has no train_data.")
    validate_num_examples(num_examples)

    if dataset_is_batched(train_data):
        train_data = Unbatch(train_data)

    if shuffle:
        buffer_size = shuffle_buffer_size or finite_dataset_len(train_data)
        if buffer_size is None:
            raise ValueError("shuffle_buffer_size is required when train_data length is unknown.")
        train_data = Shuffle(train_data, buffer_size, seed=shuffle_seed)

    if num_examples is not None:
        train_data = Take(train_data, num_examples)

    return train_data


def advance_train_state(exp, *, epochs: int = 0, steps: int = 0, phase: str = TrainState.trained):
    if epochs:
        exp.state.advance_epoch(epochs)
    if steps:
        exp.state.advance_step(steps)
    exp.state.phase = phase


def signature_discovery(obj: Any, **kwargs):
    try:
        from .tf.utils import tf_signature_discovery
        return tf_signature_discovery(obj, **kwargs)
    except (ImportError, ModuleNotFoundError):
        pass

    raise ValueError("Unable to guess a signature based on the object.")


__all__ = [
    "advance_train_state",
    "dataset_is_batched",
    "finite_dataset_len",
    "prepare_training_data",
    "signature_discovery",
    "validate_num_examples",
]
