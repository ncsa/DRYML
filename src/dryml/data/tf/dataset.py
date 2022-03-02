from dryml.data import DryData, NotIndexedError, NotSupervisedError
from typing import Callable
import tensorflow as tf


class TFDataset(DryData):
    def __init__(
            self, in_ds: tf.data.Dataset, indexed=False,
            supervised=False, batch_size=None):
        super().__init__(
            indexed=indexed, supervised=supervised,
            batch_size=batch_size)
        self.ds = in_ds

    def index(self, start=0):
        """
        If indexed, return the index of this dataset
        """
        if not self.indexed():
            raise NotIndexedError()

        return self.ds.map(
            lambda t: t[0],
            num_parallel_calls=tf.data.AUTOTUNE)

    def as_indexed(self) -> DryData:
        """
        If not already indexed, return a version of this dataset
        which is indexed.
        """
        if self.indexed():
            return self
        else:
            return TFDataset(
                self.ds.enumerate(start=0),
                indexed=True,
                supervised=self.supervised(),
                batch_size=self.batch_size())

    def as_not_indexed(self):
        """
        Strip index from dataset
        """
        if not self.indexed():
            return self
        else:
            return TFDataset(
                self.ds.map(
                    lambda t: t[1],
                    num_parallel_calls=tf.data.AUTOTUNE),
                indexed=False,
                supervised=self.supervised(),
                batch_size=self.batch_size())

    def as_not_supervised(self) -> DryData:
        """
        Strip supervised targets
        """

        if not self.supervised():
            return self
        else:
            if self.indexed():
                return TFDataset(self.ds.map(
                        lambda i, xy: (i, xy[0]),
                        num_parallel_calls=tf.data.AUTOTUNE),
                    indexed=self.indexed(),
                    supervised=False,
                    batch_size=self.batch_size())
            else:
                return TFDataset(self.ds.map(
                        lambda x, y: x,
                        num_parallel_calls=tf.data.AUTOTUNE),
                    indexed=self.indexed(),
                    supervised=False,
                    batch_size=self.batch_size())

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
        if self.batched():
            return self
        else:
            return TFDataset(
                self.ds.batch(batch_size=batch_size),
                indexed=self.indexed(),
                supervised=self.supervised(),
                batch_size=batch_size)

    def unbatch(self) -> DryData:
        """
        Unbatch this data
        """
        if not self.batched():
            return self
        else:
            return TFDataset(
                self.ds.unbatch(),
                indexed=self.indexed(),
                supervised=self.supervised())

    def apply_X(self, func: Callable = None) -> DryData:
        """
        Apply a function to the X component of DryData
        """

        if self.indexed():
            if self.supervised():
                return TFDataset(
                    self.ds.map(
                        lambda i, xy: (i, (func(xy[0]), xy[1])),
                        num_parallel_calls=tf.data.AUTOTUNE),
                    indexed=self.indexed(),
                    supervised=self.supervised(),
                    batch_size=self.batch_size())
            else:
                return TFDataset(
                    self.ds.map(
                        lambda i, x: (i, func(x)),
                        num_parallel_calls=tf.data.AUTOTUNE),
                    indexed=self.indexed(),
                    supervised=self.supervised(),
                    batch_size=self.batch_size())
        else:
            if self.supervised():
                return TFDataset(
                    self.ds.map(
                        lambda x, y: (func(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE),
                    indexed=self.indexed(),
                    supervised=self.supervised(),
                    batch_size=self.batch_size())
            else:
                return TFDataset(
                    self.ds.map(
                        lambda x: func(x),
                        num_parallel_calls=tf.data.AUTOTUNE),
                    indexed=self.indexed(),
                    supervised=self.supervised(),
                    batch_size=self.batch_size())

    def apply_Y(self, func=None) -> DryData:
        """
        Apply a function to the Y component of DryData
        """

        if not self.supervised():
            raise NotSupervisedError(
                "Can't apply a function to the Y component of "
                "non supervised dataset")

        if self.indexed():
            return TFDataset(
                self.ds.map(
                    lambda i, xy: (i, (xy[0], func(xy[1]))),
                    num_parallel_calls=tf.data.AUTOTUNE),
                indexed=self.indexed(),
                supervised=self.supervised(),
                batch_size=self.batch_size())
        else:
            return TFDataset(
                self.ds.map(
                    lambda x, y: (x, func(y)),
                    num_parallel_calls=tf.data.AUTOTUNE),
                indexed=self.indexed(),
                supervised=self.supervised(),
                batch_size=self.batch_size())

    def apply(self, func=None) -> DryData:
        """
        Apply a function to (X, Y)
        """

        if not self.supervised():
            raise NotSupervisedError(
                "Can't apply a function to the Y component of "
                "non supervised dataset")

        if self.indexed():
            return TFDataset(
                self.ds.map(
                    lambda i, xy: (i, func(*xy)),
                    num_parallel_calls=tf.data.AUTOTUNE),
                indexed=self.indexed(),
                supervised=self.supervised(),
                batch_size=self.batch_size())
        else:
            return TFDataset(
                self.ds.map(
                    lambda x, y: func(x, y),
                    num_parallel_calls=tf.data.AUTOTUNE),
                indexed=self.indexed(),
                supervised=self.supervised(),
                batch_size=self.batch_size())

    def __iter__(self):
        """
        Create iterator
        """

        return iter(self.data())
