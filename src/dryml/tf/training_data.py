"""TensorFlow-native delivery for prepared DRYML training Dataset batches."""

from __future__ import annotations

from dryml.data.native import PreparedDataset
from dryml.tf.tensor_spec import output_signature


def as_training_dataset(data: PreparedDataset, preparation):
    """Expose one prepared Dataset pipeline as a native ``tf.data.Dataset``.

    Args:
        data: Dependency-light prepared Dataset pipeline.
        preparation: Retained x/y handoff whose consumer specs form the TensorFlow
            output signature.

    Returns:
        A re-iterable ``tf.data.Dataset`` yielding authored prepared ``(x, y)``
        batches without batching, shuffling, or repeating them again.

    Raises:
        TypeError: If ``data`` is not a :class:`PreparedDataset` or preparation
            does not expose TensorFlow consumer specs.
        DatasetExhaustedError: During iteration if a finite source ends early.

    Side Effects:
        Imports TensorFlow only when called. Each native traversal opens and closes
        an independent DRYML cursor; adapter reads never update training progress.
    """

    if not isinstance(data, PreparedDataset):
        raise TypeError("TensorFlow training input requires a PreparedDataset.")
    try:
        specs = preparation.consumer_specs
    except AttributeError as error:
        raise TypeError("TensorFlow training input requires prepared consumer specs.") from error
    import tensorflow as tf

    def prepared_values():
        """Yield one native traversal while guaranteeing cursor cleanup on cancellation."""

        cursor = data.training_batches(preparation)
        try:
            yield from cursor
        finally:
            cursor.close()

    return tf.data.Dataset.from_generator(
        prepared_values,
        output_signature=output_signature(specs),
    )


def iter_training_batches(data: PreparedDataset, preparation):
    """Return one closeable prepared TensorFlow batch cursor for explicit loops.

    Args:
        data: Dependency-light prepared Dataset pipeline.
        preparation: Retained x/y handoff prepared for TensorFlow.

    Returns:
        A closeable iterator yielding native TensorFlow ``(x, y)`` batches.

    Raises:
        TypeError: If ``data`` is not a :class:`PreparedDataset`.
        DatasetExhaustedError: During iteration if a finite source ends early.

    Side Effects:
        Opens no source until the cursor is advanced and never changes training
        progress, accepted-update state, or checkpoint association.
    """

    if not isinstance(data, PreparedDataset):
        raise TypeError("TensorFlow training input requires a PreparedDataset.")
    return data.training_batches(preparation)


__all__ = ["as_training_dataset", "iter_training_batches"]
