from dryml.data import Dataset, \
    NumpyDataset, util
from typing import Callable
import torch
import numpy as np
from dryml.data.util import taker, skiper, nested_batcher


class TorchIterableDatasetWrapper(torch.utils.data.IterableDataset):
    def __init__(
            self, iterable_gen: Callable):
        self.iterable_gen = iterable_gen

    def __iter__(self):
        return iter(self.iterable_gen())


class TorchDataset(Dataset):
    def __init__(
            self, in_ds: torch.utils.data.Dataset, indexed=False,
            supervised=False, batch_size=None, size=None):

        super().__init__(
            indexed=indexed, supervised=supervised,
            batch_size=batch_size)
        self.ds = in_ds
        if size is None:
            size = np.nan
        self.size = size

    def as_indexed(self, start=0) -> Dataset:
        """
        If not already indexed, return a version of this dataset
        which is indexed.
        """
        if self.indexed:
            return self
        else:
            if not self.batched:
                def enumerate_dataset(gen_func, start=0):
                    i = start
                    it = iter(gen_func())
                    while True:
                        try:
                            d = next(it)
                        except StopIteration:
                            return
                        yield (torch.tensor(i), d)
                        i += 1

                torch_ds = TorchIterableDatasetWrapper(
                    lambda: enumerate_dataset(self.data_gen, start=start))

                return TorchDataset(
                    torch_ds,
                    indexed=True,
                    supervised=self.supervised,
                    batch_size=self.batch_size,
                    size=self.size)
            else:
                def enumerate_dataset(gen_func, start=0):
                    it = iter(gen_func())
                    i = start
                    while True:
                        try:
                            d = next(it)
                        except StopIteration:
                            return
                        from dryml.data.util import get_data_batch_size
                        batch_size = get_data_batch_size(d)
                        idx = torch.tensor(list(range(i, i+batch_size)))
                        i += batch_size
                        yield (idx, d)

                torch_ds = TorchIterableDatasetWrapper(
                    lambda: enumerate_dataset(
                        lambda: iter(self.ds),
                        start=start))

                return TorchDataset(
                    torch_ds,
                    indexed=True,
                    supervised=self.supervised,
                    batch_size=self.batch_size,
                    size=self.size)

    @property
    def data_gen(self):
        return lambda: self.ds

    def data(self):
        """
        Get the internal dataset
        """
        return self.ds

    def batch(self, batch_size=32, drop_remainder=True) -> Dataset:
        """
        Batch this data
        """
        if self.batched:
            if self.batch_size != batch_size:
                return self.unbatch().batch(batch_size=batch_size)
            else:
                return self
        else:

            obj = TorchIterableDatasetWrapper(
                lambda: nested_batcher(
                    self.data_gen, batch_size,
                    lambda e: torch.stack(e, dim=0),
                    drop_remainder=drop_remainder))

            return TorchDataset(
                obj,
                indexed=self.indexed,
                supervised=self.supervised,
                batch_size=batch_size,
                size=self.size)

    def unbatch(self) -> Dataset:
        """
        Unbatch this data
        """
        if not self.batched:
            return self
        else:
            obj = TorchIterableDatasetWrapper(
                lambda: util.nested_unbatcher(self.data_gen))
            return TorchDataset(
                obj,
                indexed=self.indexed,
                supervised=self.supervised,
                size=self.size)

    def map(self, func: Callable = None):
        obj = TorchIterableDatasetWrapper(
            lambda: map(func, self.data_gen()))

        return TorchDataset(
            obj,
            indexed=self.indexed,
            supervised=self.supervised,
            batch_size=self.batch_size,
            size=self.size)

    def take(self, n):
        """
        Take only a specific number of examples
        """

        new_size = self.size
        if new_size is np.nan:
            new_size = n
        elif new_size is np.inf:
            new_size = n
        else:
            if new_size > n:
                new_size = n

        obj = TorchIterableDatasetWrapper(
            lambda: taker(self.data_gen, n))

        return TorchDataset(
            obj,
            indexed=self.indexed,
            supervised=self.supervised,
            batch_size=self.batch_size,
            size=new_size)

    def skip(self, n):
        """
        Skip a specific number of examples
        """

        new_size = self.size
        if new_size is not np.nan and new_size is not np.inf:
            if n > new_size:
                new_size = 0
            else:
                new_size -= n

        obj = TorchIterableDatasetWrapper(
            lambda: skiper(self.data_gen, n))

        return TorchDataset(
            obj,
            indexed=self.indexed,
            supervised=self.supervised,
            batch_size=self.batch_size,
            size=new_size)

    def __len__(self):
        """
        Get length of dataset. Will return Infinite if infinite,
        and unknown if it can't be determined.
        """
        return self.size

    def numpy(self):
        dataset = self.map_el(lambda el: el.detach().cpu().numpy())
        return NumpyDataset(
            dataset.data_gen,
            indexed=dataset.indexed,
            supervised=dataset.supervised,
            batch_size=dataset.batch_size,
            size=dataset.size)

    def tf(self):
        return self.numpy().tf()

    def torch(self):
        return self

    def shuffle(self, buffer_size, seed=None):
        # create generator of data
        def shuffler():
            # Create new generator
            rng = np.random.default_rng(seed=seed)

            # create iterator on unbatched data
            ds_iter = iter(self.unbatch())

            # create buffer and fill it
            el_buffer = []

            # Fill the buffer
            for idx in range(buffer_size):
                try:
                    el_buffer.append(next(ds_iter))
                except StopIteration:
                    break

            while True:
                # Compute size of buffer, and break out if nothing left
                num_in_buffer = len(el_buffer)
                if num_in_buffer == 0:
                    break

                if num_in_buffer > 1:
                    idx = rng.integers(low=0, high=num_in_buffer-1)
                else:
                    idx = 0

                yield el_buffer.pop(idx)

                # Try to refill the buffer here
                try:
                    el_buffer.append(next(ds_iter))
                except StopIteration:
                    pass

        ds = TorchIterableDatasetWrapper(
            shuffler)

        return TorchDataset(
            ds,
            indexed=self.indexed,
            supervised=self.supervised,
            batch_size=None)
