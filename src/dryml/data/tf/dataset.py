from dryml.data import Dataset, \
    NumpyDataset, util
from typing import Callable
import tensorflow as tf
import numpy as np


class TFDataset(Dataset):
    def __init__(
            self, in_ds: tf.data.Dataset, indexed=False,
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
            return TFDataset(
                self.ds.enumerate(start=start),
                indexed=True,
                supervised=self.supervised,
                batch_size=self.batch_size)

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
            if self.batch_size == batch_size:
                return self
            else:
                return self.unbatch().batch(batch_size=batch_size)
        else:
            return TFDataset(
                self.ds.batch(
                    batch_size=batch_size,
                    drop_remainder=drop_remainder),
                indexed=self.indexed,
                supervised=self.supervised,
                batch_size=batch_size)

    def unbatch(self) -> Dataset:
        """
        Unbatch this data
        """
        if not self.batched:
            return self
        else:
            return TFDataset(
                self.ds.unbatch(),
                indexed=self.indexed,
                supervised=self.supervised)

    def map(self, func: Callable = None) -> Dataset:
        """
        Apply a function to the X component of Dataset
        """

        return TFDataset(
            self.ds.map(
                lambda *t: func(t),
                num_parallel_calls=tf.data.AUTOTUNE),
            indexed=self.indexed,
            supervised=self.supervised,
            batch_size=self.batch_size,
            size=self.size)

    def take(self, n):
        """
        Take only a specific number of examples
        """
        return TFDataset(
            self.ds.take(n),
            indexed=self.indexed,
            supervised=self.supervised,
            batch_size=self.batch_size)

    def skip(self, n):
        """
        Skip a specific number of examples
        """
        return TFDataset(
            self.ds.skip(n),
            indexed=self.indexed,
            supervised=self.supervised,
            batch_size=self.batch_size)

    def __len__(self):
        """
        Get length of dataset. Will return Infinite if infinite,
        and unknown if it can't be determined.
        """
        cardinality = self.ds.cardinality()
        if cardinality == tf.data.INFINITE_CARDINALITY:
            return np.inf
        if cardinality == tf.data.UNKNOWN_CARDINALITY:
            return np.nan
        return cardinality.numpy()

    def numpy(self):
        """
        Produce NumpyDataset from this TFDataset
        """

        def numpy_transform(el):
            if type(el) is not np.ndarray:
                return el.numpy()
            else:
                return el

        def numpy_generator():
            return map(
                util.nestize(numpy_transform),
                self.data_gen())

        return NumpyDataset(
            numpy_generator,
            indexed=self.indexed,
            supervised=self.supervised,
            batch_size=self.batch_size)

    def tf(self):
        """
        Get a TFDataset, but this is already a TFDataset
        """
        return self

    def torch(self):
        """
        Create TorchDataset from this dataset
        """

        import torch
        from dryml.data.torch import TorchDataset, TorchIterableDatasetWrapper

        def tf_to_torch(el):
            return torch.tensor(el.numpy())

        obj = TorchIterableDatasetWrapper(
            lambda: map(util.nestize(tf_to_torch), self.data_gen()))

        return TorchDataset(
            obj,
            indexed=self.indexed,
            supervised=self.supervised,
            batch_size=self.batch_size,
            size=self.size)

    def shuffle(self, buffer_size, seed=None):
        return TFDataset(
            self.ds.shuffle(buffer_size, seed=seed),
            indexed=self.indexed,
            supervised=self.supervised,
            batch_size=self.batch_size)
