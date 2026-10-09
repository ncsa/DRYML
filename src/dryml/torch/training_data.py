"""PyTorch-native delivery for prepared DRYML training Dataset batches."""

from __future__ import annotations

from dryml.data.native import PreparedDataset


def as_training_dataset(data: PreparedDataset, preparation):
    """Expose a prepared pipeline as a native single-process ``IterableDataset``.

    Args:
        data: Dependency-light prepared Dataset pipeline.
        preparation: Retained x/y handoff prepared for PyTorch.

    Returns:
        A PyTorch ``IterableDataset`` whose every ``__iter__`` opens a fresh
        DRYML cursor and yields already-batched ``(x, y)`` values.

    Raises:
        TypeError: If ``data`` is not a :class:`PreparedDataset`.

    Side Effects:
        Imports PyTorch only when called. The wrapper has no worker partitioning,
        prefetching, batching, or progress/checkpoint side effects.
    """

    if not isinstance(data, PreparedDataset):
        raise TypeError("Torch training input requires a PreparedDataset.")
    import torch

    class PreparedIterableDataset(torch.utils.data.IterableDataset):
        """Native IterableDataset reopening the prepared DRYML pipeline per traversal."""

        def __iter__(self):
            """Yield one independent traversal and close it when consumption stops."""

            cursor = data.training_batches(preparation)
            try:
                yield from cursor
            finally:
                cursor.close()

    return PreparedIterableDataset()


def as_data_loader(dataset):
    """Build the supported one-process DataLoader for already-batched native data.

    Args:
        dataset: Native IterableDataset returned by :func:`as_training_dataset`.

    Returns:
        A ``torch.utils.data.DataLoader`` with ``num_workers=0`` and
        ``batch_size=None`` so it neither duplicates traversal nor adds a batch axis.

    Raises:
        TypeError: If ``dataset`` is not a native PyTorch IterableDataset.

    Side Effects:
        Imports PyTorch only when called. DataLoader consumption advances only its
        private DRYML cursor; successful training updates remain progress owners.
    """

    import torch

    if not isinstance(dataset, torch.utils.data.IterableDataset):
        raise TypeError("Torch DataLoader input must be an IterableDataset.")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        collate_fn=lambda value: value,
    )


def iter_training_batches(data: PreparedDataset, preparation):
    """Return one closeable prepared Torch batch cursor for explicit loops.

    Args:
        data: Dependency-light prepared Dataset pipeline.
        preparation: Retained x/y handoff prepared for PyTorch.

    Returns:
        A closeable iterator yielding native Torch ``(x, y)`` batches.

    Raises:
        TypeError: If ``data`` is not a :class:`PreparedDataset`.
        DatasetExhaustedError: During iteration if a finite source ends early.

    Side Effects:
        Opens no source until advancement and never changes retained training
        progress, exposure, or checkpoint association.
    """

    if not isinstance(data, PreparedDataset):
        raise TypeError("Torch training input requires a PreparedDataset.")
    return data.training_batches(preparation)


__all__ = ["as_data_loader", "as_training_dataset", "iter_training_batches"]
