from dryml.data import DryData, NotIndexedError, NotSupervisedError, \
    NumpyDataset, util
from typing import Callable
import tensorflow as tf
import numpy as np


class TFDataset(DryData):
    def __init__(
            self, in_ds: tf.data.Dataset, indexed=False,
            supervised=False, batch_size=None):
        super().__init__(
            indexed=indexed, supervised=supervised,
            batch_size=batch_size)
        self.ds = in_ds

    def index(self):
        """
        If indexed, return the index of this dataset
        """
        if not self.indexed:
            raise NotIndexedError()

        return self.ds.map(
            lambda t: t[0],
            num_parallel_calls=tf.data.AUTOTUNE)

    def as_indexed(self, start=0) -> DryData:
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

    def as_not_indexed(self):
        """
        Strip index from dataset
        """
        if not self.indexed:
            return self
        else:
            return TFDataset(
                self.ds.map(
                    lambda t: t[1],
                    num_parallel_calls=tf.data.AUTOTUNE),
                indexed=False,
                supervised=self.supervised,
                batch_size=self.batch_size)

    def as_not_supervised(self) -> DryData:
        """
        Strip supervised targets
        """

        if not self.supervised:
            return self
        else:
            if self.indexed:
                return TFDataset(self.ds.map(
                        lambda i, xy: (i, xy[0]),
                        num_parallel_calls=tf.data.AUTOTUNE),
                    indexed=self.indexed,
                    supervised=False,
                    batch_size=self.batch_size)
            else:
                return TFDataset(self.ds.map(
                        lambda x, y: x,
                        num_parallel_calls=tf.data.AUTOTUNE),
                    indexed=self.indexed,
                    supervised=False,
                    batch_size=self.batch_size)

    def intersect(self) -> DryData:
        """
        Intersect this dataset with another
        """
        raise NotImplementedError()

    def data(self):
        """
        Get the internal dataset
        """
        return self.ds

    def batch(self, batch_size=32) -> DryData:
        """
        Batch this data
        """
        if self.batched:
            return self
        else:
            return TFDataset(
                self.ds.batch(batch_size=batch_size),
                indexed=self.indexed,
                supervised=self.supervised,
                batch_size=batch_size)

    def unbatch(self) -> DryData:
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

    def apply_X(self, func: Callable = None) -> DryData:
        """
        Apply a function to the X component of DryData
        """

        if self.indexed:
            if self.supervised:
                return TFDataset(
                    self.ds.map(
                        lambda i, xy: (i, (func(xy[0]), xy[1])),
                        num_parallel_calls=tf.data.AUTOTUNE),
                    indexed=self.indexed,
                    supervised=self.supervised,
                    batch_size=self.batch_size)
            else:
                return TFDataset(
                    self.ds.map(
                        lambda i, x: (i, func(x)),
                        num_parallel_calls=tf.data.AUTOTUNE),
                    indexed=self.indexed,
                    supervised=self.supervised,
                    batch_size=self.batch_size)
        else:
            if self.supervised:
                return TFDataset(
                    self.ds.map(
                        lambda x, y: (func(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE),
                    indexed=self.indexed,
                    supervised=self.supervised,
                    batch_size=self.batch_size)
            else:
                return TFDataset(
                    self.ds.map(
                        lambda x: func(x),
                        num_parallel_calls=tf.data.AUTOTUNE),
                    indexed=self.indexed,
                    supervised=self.supervised,
                    batch_size=self.batch_size)

    def apply_Y(self, func=None) -> DryData:
        """
        Apply a function to the Y component of DryData
        """

        if not self.supervised:
            raise NotSupervisedError(
                "Can't apply a function to the Y component of "
                "non supervised dataset")

        if self.indexed:
            return TFDataset(
                self.ds.map(
                    lambda i, xy: (i, (xy[0], func(xy[1]))),
                    num_parallel_calls=tf.data.AUTOTUNE),
                indexed=self.indexed,
                supervised=self.supervised,
                batch_size=self.batch_size)
        else:
            return TFDataset(
                self.ds.map(
                    lambda x, y: (x, func(y)),
                    num_parallel_calls=tf.data.AUTOTUNE),
                indexed=self.indexed,
                supervised=self.supervised,
                batch_size=self.batch_size)

    def apply(self, func=None) -> DryData:
        """
        Apply a function to (X, Y)
        """

        if not self.supervised:
            raise NotSupervisedError(
                "Can't apply a function to the Y component of "
                "non supervised dataset")

        if self.indexed:
            return TFDataset(
                self.ds.map(
                    lambda i, xy: (i, func(*xy)),
                    num_parallel_calls=tf.data.AUTOTUNE),
                indexed=self.indexed,
                supervised=self.supervised,
                batch_size=self.batch_size)
        else:
            return TFDataset(
                self.ds.map(
                    lambda x, y: func(x, y),
                    num_parallel_calls=tf.data.AUTOTUNE),
                indexed=self.indexed,
                supervised=self.supervised,
                batch_size=self.batch_size)

    def __iter__(self):
        """
        Create iterator
        """

        return iter(self.data())

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

        def numpy_transformer(gen):
            it = iter(gen)
            while True:
                try:
                    el = next(it)
                except StopIteration:
                    break
                yield util.nested_apply(el, numpy_transform)

            return

        return NumpyDataset(
            numpy_transformer(self.ds),
            indexed=self.indexed,
            supervised=self.supervised,
            batch_size=self.batch_size)

    def tf(self):
        """
        Get a TFDataset, but this is already a TFDataset
        """
        return self
