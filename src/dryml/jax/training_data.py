"""JAX-native batch iteration for prepared DRYML Dataset pipelines."""

from __future__ import annotations

from dryml.data.native import BatchPreparation, NativeTrainingCursor, PreparedDataset
from dryml.data.traversal import EpochView


def iter_training_batches(data: PreparedDataset | EpochView, preparation: BatchPreparation,
                          *, epoch: int = 0) -> NativeTrainingCursor:
    """Return one closeable iterator yielding JAX-native prepared ``(x, y)`` batches.

    Args:
        data: PreparedDataset independent pass or borrowed finite EpochView.
        preparation: Retained x/y handoff prepared for JAX.
        epoch: Independent-pass ordinal; must be zero for a borrowed view.

    Returns:
        A closeable cursor whose yielded trees retain authored structure, dtype,
        shape, batch axis, and order.

    Raises:
        TypeError: If ``data`` is not PreparedDataset or EpochView.
        ValueError: Nonzero epoch supplied with a borrowed view.
        DatasetExhaustedError: During iteration if a finite source ends early.

    Side Effects:
        Opens no source until advancement. The selected preparation may import JAX
        to execute its retained conversion edge, but cursor reads never update
        training progress, exposure, or checkpoint state.
        Use the cursor as a context manager around updates to preserve consumer
        failures over cleanup errors; closure alone does not prove output EOF.
    """

    if isinstance(data, EpochView):
        if type(epoch) is not int or epoch != 0:
            raise ValueError("A borrowed view already selects its epoch.")
        return data.training_batches(preparation)
    if not isinstance(data, PreparedDataset):
        raise TypeError("JAX training input requires a PreparedDataset or EpochView.")
    return data.training_batches(preparation, epoch=epoch)


__all__ = ["iter_training_batches"]
