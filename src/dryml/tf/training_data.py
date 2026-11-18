"""TensorFlow-native delivery for prepared DRYML training Dataset batches."""

from __future__ import annotations

from dryml.data.native import BatchPreparation, NativeTrainingCursor, PreparedDataset
from dryml.data.traversal import EpochView
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

        with data.training_batches(preparation) as cursor:
            yield from cursor

    return tf.data.Dataset.from_generator(
        prepared_values,
        output_signature=output_signature(specs),
    )


def iter_training_batches(data: PreparedDataset | EpochView, preparation: BatchPreparation,
                          *, epoch: int = 0) -> NativeTrainingCursor:
    """Return one closeable prepared TensorFlow batch cursor for explicit loops.

    Args:
        data: PreparedDataset independent pass or borrowed finite EpochView.
        preparation: Retained x/y handoff prepared for TensorFlow.
        epoch: Independent-pass ordinal; must be zero for a borrowed view.

    Returns:
        A closeable iterator yielding native TensorFlow ``(x, y)`` batches.

    Raises:
        TypeError: If ``data`` is not PreparedDataset or EpochView.
        ValueError: Nonzero epoch supplied with a borrowed view.
        DatasetExhaustedError: During iteration if a finite source ends early.

    Side Effects:
        Opens no source until the cursor is advanced and never changes training
        progress, accepted-update state, or checkpoint association.
        Use the cursor as a context manager around updates to preserve consumer
        failures over cleanup errors; closure alone does not prove output EOF.
    """

    if isinstance(data, EpochView):
        if type(epoch) is not int or epoch != 0:
            raise ValueError("A borrowed view already selects its epoch.")
        return data.training_batches(preparation)
    if not isinstance(data, PreparedDataset):
        raise TypeError("TensorFlow training input requires a PreparedDataset or EpochView.")
    return data.training_batches(preparation, epoch=epoch)


__all__ = ["as_training_dataset", "iter_training_batches"]
