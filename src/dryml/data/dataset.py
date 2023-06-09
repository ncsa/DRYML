from __future__ import annotations
from typing import Callable
from dryml.data.util import nestize


class NotIndexedError():
    pass


class NotSupervisedError():
    pass


class Dataset(object):
    """
    A Simple wrapper class to house data operations
    """

    def __init__(self, indexed=False, supervised=False,
                 batch_size=None):
        self._indexed = indexed
        self._supervised = supervised
        self._batch_size = batch_size

    @property
    def indexed(self) -> bool:
        """
        Indicate whether this dataset is indexed.
        """
        return self._indexed

    def index(self):
        """
        If indexed, return the index of this dataset
        """
        if not self.indexed:
            raise NotIndexedError()

        return self.map(lambda t: t[0])

    def as_indexed(self, start=0) -> Dataset:
        """
        If not already indexed, return a version of this dataset
        which is indexed.
        """
        raise NotImplementedError()

    def as_not_indexed(self):
        """
        Strip index from dataset
        """
        if not self.indexed:
            return self
        else:
            return self.map(lambda t: t[1])

    @property
    def supervised(self) -> bool:
        """
        Indicate whether this dataset is supervised (has targets as well)
        """
        return self._supervised

    def as_not_supervised(self) -> Dataset:
        """
        Strip supervised targets
        """

        if not self.supervised:
            return self
        else:
            if self.indexed:
                return self.map(lambda t: (t[0], t[1][0]))
            else:
                return self.map(lambda t: t[0])

    def intersect(self) -> Dataset:
        """
        Intersect this dataset with another
        """
        raise NotImplementedError()

    @property
    def data_gen(self):
        """
        Gives a function where calling it returns a generator of the dataset.
        """
        raise NotImplementedError()

    def data(self):
        """
        Return the backing dataset
        """
        raise NotImplementedError()

    @property
    def batched(self) -> bool:
        """
        Indicate whether this data has been batched
        """
        if self._batch_size is not None:
            return True
        else:
            return False

    @property
    def batch_size(self):
        """
        Get the batch size
        """

        return self._batch_size

    def batch(self, batch_size=32) -> Dataset:
        """
        Batch this data
        """
        raise NotImplementedError()

    def unbatch(self) -> Dataset:
        """
        Unbatch this data
        """
        raise NotImplementedError()

    def map(self, func: Callable = None) -> Dataset:
        """
        Apply a function to the data of Dataset
        """
        raise NotImplementedError()

    def map_el(self, func: Callable = None) -> Dataset:
        """
        Apply a function to every element in Dataset, even nesting in
        """

        return self.map(nestize(func))

    def apply_X(
            self,
            func: Callable = None,
            func_args=(),
            func_kwargs={}) -> Dataset:
        """
        Apply a function to the X component of Dataset

        Args:
            func: The function to apply
            func_args: Arguments to pass to the function
            func_kwargs: Keyword arguments to pass to the function
        """

        if self.indexed:
            if self.supervised:
                return self.map(
                    lambda t: (
                        t[0],
                        (func(t[1][0], *func_args, **func_kwargs),
                         t[1][1])
                    )
                )
            else:
                return self.map(
                    lambda t: (
                        t[0],
                        func(t[1], *func_args, **func_kwargs)
                    )
                )
        else:
            if self.supervised:
                return self.map(
                    lambda t: (
                        func(t[0], *func_args, **func_kwargs),
                        t[1]
                    )
                )
            else:
                return self.map(
                    lambda x: func(x, *func_args, **func_kwargs)
                )

    def apply_Y(
            self,
            func: Callable = None,
            func_args=(),
            func_kwargs={}) -> Dataset:
        """
        Apply a function to the Y component of Dataset

        Args:
            func: The function to apply
            func_args: Arguments to pass to the function
            func_kwargs: Keyword arguments to pass to the function
        """

        if not self.supervised:
            raise NotSupervisedError(
                "Can't apply a function to the Y component of "
                "non supervised dataset")

        if self.indexed:
            return self.map(
                lambda t: (
                    t[0],
                    (t[1][0],
                     func(t[1][1], *func_args, **func_kwargs))
                )
            )
        else:
            return self.map(
                lambda t: (
                    t[0],
                    func(t[1], *func_args, **func_kwargs)
                )
            )

    def apply(
            self,
            func: Callable = None,
            func_args=(),
            func_kwargs={}) -> Dataset:
        """
        Apply a function to (X, Y)

        Args:
            func: The function to apply
            func_args: Arguments to pass to the function
            func_kwargs: Keyword arguments to pass to the function
        """

        if not self.supervised:
            raise NotSupervisedError(
                "Can't apply a function to the Y component of "
                "non supervised dataset")

        if self.indexed:
            return self.map(
                lambda t: (
                    t[0],
                    func(*t[1], *func_args, **func_kwargs)
                )
            )
        else:
            return self.map(
                lambda t: func(*t, *func_args, **func_kwargs)
            )

    def __iter__(self):
        """
        Create iterator
        """

        return iter(self.data())

    def take(self, n):
        """
        Take only a specific number of examples
        """
        raise NotImplementedError()

    def skip(self, n):
        """
        Skip a specific number of examples
        """
        raise NotImplementedError()

    def __len__(self):
        """
        Get length of dataset. Will return Infinite if infinite,
        and unknown if it can't be determined.
        """
        raise NotImplementedError()

    def numpy(self):
        """
        Create a NumpyDataset from this dataset.
        """

        raise NotImplementedError()

    def tf(self):
        """
        Create a TFDataset from this dataset.
        """

        raise NotImplementedError()

    def collect(self):
        """
        Collect all data from the dataset
        """

        result = []
        for el in self:
            result.append(el)
        return result

    def peek(self):
        """
        Get the first element to have a look
        """

        item_list = self.take(1).collect()

        if len(item_list) == 0:
            raise RuntimeError(
                "Can't peek, no data in dataset. If you expect data, "
                "double check there isn't a batch function called "
                "with drop_remainder=True.")

        return item_list[0]

    def count(self, limit=-1):
        """
        Attempt to count 'elements' in the Dataset
        """

        number = 0
        for e in self:
            number += 1
            if limit > 0:
                if number > limit:
                    break
        return number

    def shuffle(self, buffer_size, seed=None):
        """
        Shuffle elements of dataset.
        """

        raise NotImplementedError()
